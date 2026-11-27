"""Collection commune texte/image, persistance et recherches CLIP (support 36–46)."""
import base64
import hashlib
import io
import json
import re
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from PIL import Image, ImageOps

from .config import EMBEDDING, FINGERPRINT, Settings, ecrire_json, lire_json
from .embeddings import get_encoder
from .extraction import extraire


def reference(identifiant):
    return "S" + hashlib.sha256(identifiant.encode()).hexdigest()[:12]


def image_base64(path, max_size=(900, 900)):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail(max_size)
        buffer = io.BytesIO()
        im.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class RagStore:
    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.client = chromadb.PersistentClient(path=str(self.settings.data / "chroma"),
                    settings=ChromaSettings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name="pdf_clip_" + FINGERPRINT, embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_config": json.dumps(EMBEDDING, sort_keys=True)})
        if self.collection.metadata.get("embedding_config") != json.dumps(EMBEDDING, sort_keys=True):
            raise ValueError("La collection appartient à une autre configuration d’embedding.")
        self.manifest_path = self.settings.data / "documents.json"

    @property
    def encoder(self):
        return get_encoder(str(self.settings.data), self.settings.clip_device)

    def indexer(self, pdf: Path, alias: str | None = None, force=False):
        debut = time.perf_counter()
        extracted = extraire(pdf, self.settings)
        doc = extracted["document_id"]
        alias = alias or Path(pdf).name
        documents = lire_json(self.manifest_path, {})
        ancien = documents.get(alias)
        if not force and ancien and ancien["document_id"] == doc and self.collection.count() > 0:
            total = len(self.collection.get(where={"document_id": doc})["ids"])
            if total == ancien["total"]:
                return {**ancien, "reutilise": True, "duree_secondes": time.perf_counter() - debut}
        ids, textes, metas, tokens = [], [], [], []
        existants = self.collection.get(where={"document_id": doc}, include=["embeddings"])
        anciens_vecteurs = {ident: emb for ident, emb in zip(existants["ids"],
            existants["embeddings"] if existants["embeddings"] is not None else [])}
        for item in extracted["textes"]:
            texte = self.settings.chemin(item["source"]).read_text(encoding="utf-8")
            for fragment, chunk in enumerate(self.encoder.decouper_texte(texte)):
                ids.append(f"{doc}:texte:{item['page']}:{fragment}")
                textes.append(chunk["texte"])
                tokens.append(chunk["tokens"])
                metas.append({"document_id": doc, "pdf": extracted["pdf"], "type": "texte",
                              "page": item["page"], "fragment": fragment, "source": item["source"],
                              "tokens": len(chunk["tokens"]), "debut_token": chunk["debut_token"],
                              "embedding_config": FINGERPRINT})
        for debut_lot in range(0, len(ids), 32):
            fin = debut_lot + 32
            lot_ids = ids[debut_lot:fin]
            manquants = [i for i, ident in enumerate(lot_ids) if ident not in anciens_vecteurs]
            if manquants:
                nouveaux = self.encoder.encoder_tokens([tokens[debut_lot + i] for i in manquants])
                anciens_vecteurs.update({lot_ids[i]: vec for i, vec in zip(manquants, nouveaux)})
            vecteurs = [anciens_vecteurs[ident].tolist() for ident in lot_ids]
            self.collection.upsert(ids=ids[debut_lot:fin], documents=textes[debut_lot:fin],
                                   metadatas=metas[debut_lot:fin], embeddings=vecteurs)
        n_texte = len(ids)
        for debut_lot in range(0, len(extracted["images"]), 8):
            lot = extracted["images"][debut_lot:debut_lot + 8]
            image_ids = [f"{doc}:image:{x['page']}:{x['sha256']}" for x in lot]
            manquants = [i for i, ident in enumerate(image_ids) if ident not in anciens_vecteurs]
            if manquants:
                nouveaux = self.encoder.encoder_images([self.settings.chemin(lot[i]["source"]) for i in manquants])
                anciens_vecteurs.update({image_ids[i]: vec for i, vec in zip(manquants, nouveaux)})
            self.collection.upsert(ids=image_ids, embeddings=[anciens_vecteurs[ident].tolist() for ident in image_ids],
                documents=[f"Figure {x['figure']}" if x['figure'] else "Image intégrée" for x in lot],
                metadatas=[{"document_id": doc, "pdf": extracted["pdf"], "type": "image",
                    "page": x["page"], "fragment": -1, "source": x["source"], "figure": x["figure"],
                    "methode": x["methode"], "embedding_config": FINGERPRINT} for x in lot])
            ids.extend(image_ids)
        # Supprimer les éléments devenus obsolètes seulement après les upserts réussis.
        presents = set(self.collection.get(where={"document_id": doc})["ids"])
        obsoletes = list(presents - set(ids))
        if obsoletes:
            self.collection.delete(ids=obsoletes)
        if ancien and ancien["document_id"] != doc:
            autres = [v["document_id"] for k, v in documents.items() if k != alias]
            if ancien["document_id"] not in autres:
                self.collection.delete(where={"document_id": ancien["document_id"]})
        bilan = {"document_id": doc, "pdf": extracted["pdf"], "pages": extracted["pages"],
                 "chunks": n_texte, "images": len(extracted["images"]), "total": len(ids),
                 "embedding": EMBEDDING, "reutilise": False}
        documents[alias] = bilan
        ecrire_json(self.manifest_path, documents)
        return {**bilan, "duree_secondes": time.perf_counter() - debut}

    def register_image(self, contenu: bytes):
        """Enregistrer une image fournie par l’utilisateur, jamais un chemin du modèle."""
        if len(contenu) > 15 * 1024 * 1024:
            raise ValueError("Image trop volumineuse (15 Mo maximum).")
        with Image.open(io.BytesIO(contenu)) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.width * im.height > 30_000_000:
                raise ValueError("Image trop grande.")
            im.thumbnail((1800, 1800))
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        identifiant = hashlib.sha256(buf.getvalue()).hexdigest()
        cible = self.settings.data / "queries" / (identifiant + ".png")
        cible.parent.mkdir(exist_ok=True)
        cible.write_bytes(buf.getvalue())
        return identifiant

    def query_image(self, identifiant):
        if not isinstance(identifiant, str) or not re.fullmatch(r"[a-f0-9]{64}", identifiant):
            raise ValueError("Identifiant d’image de requête non autorisé.")
        p = self.settings.chemin("queries/" + identifiant + ".png")
        if not p.is_file():
            raise ValueError("Image de requête non enregistrée.")
        return p

    def image_source(self, identifiant):
        resultat = self.collection.get(ids=[identifiant], include=["metadatas"])
        if not resultat["ids"] or resultat["metadatas"][0]["type"] != "image":
            raise ValueError("Identifiant de source visuelle inconnu.")
        return self.settings.chemin(resultat["metadatas"][0]["source"])

    def rechercher(self, texte=None, image_id=None, type_resultat="tous", k=6, document_id=None):
        if texte is not None and (not isinstance(texte, str) or not texte.strip()):
            raise ValueError("La requête textuelle doit être non vide.")
        if texte is None and image_id is None:
            raise ValueError("Fournir un texte, une image enregistrée, ou les deux.")
        if type_resultat not in {"texte", "image", "tous"}:
            raise ValueError("Modalité inconnue.")
        if type(k) is not int or not 1 <= k <= 12:
            raise ValueError("k doit être un entier entre 1 et 12.")
        image_path = self.query_image(image_id) if image_id is not None else None
        conditions = []
        if type_resultat != "tous":
            conditions.append({"type": type_resultat})
        numero_figure = re.search(r"\bfigure\s+(\d+)\b", texte or "", re.I) if type_resultat == "image" else None
        if numero_figure:
            conditions.append({"figure": int(numero_figure.group(1))})
        if document_id:
            if not re.fullmatch(r"[a-f0-9]{64}", document_id):
                raise ValueError("Identifiant de document invalide.")
            conditions.append({"document_id": document_id})
        where = {"$and": conditions} if len(conditions) > 1 else (conditions[0] if conditions else None)
        options = {"where": where} if where else {}
        total = len(self.collection.get(**options, include=[])["ids"])
        if total == 0:
            return []
        vecteurs = []
        if texte is not None:
            vecteurs.append(self.encoder.encoder_textes([texte])[0])
        if image_path is not None:
            vecteurs.append(self.encoder.encoder_images([image_path])[0])
        # Fusion RRF (constante 60), sur le même univers filtré : tous les candidats
        # jusqu’à 48 par modalité, puis k résultats. Les cosinus restent séparés.
        hybride = len(vecteurs) == 2
        candidats = {}
        for index, v in enumerate(vecteurs):
            resultat = self.collection.query(query_embeddings=[v.tolist()],
                n_results=min(total, max(k, 48) if hybride else k),
                include=["documents", "metadatas", "distances"], **options)
            for rang, (ident, contenu, meta, distance) in enumerate(zip(resultat["ids"][0],
                    resultat["documents"][0], resultat["metadatas"][0], resultat["distances"][0]), 1):
                item = candidats.setdefault(ident, {"id": ident, "reference": reference(ident),
                    "contenu": contenu, "metadata": meta, "similarites": {}, "score_fusion": 0.0})
                item["similarites"]["texte" if texte is not None and index == 0 else "image"] = 1 - float(distance)
                item["score_fusion"] += 1 / (60 + rang)
        tries = sorted(candidats.values(), key=lambda x: (-x["score_fusion"], x["id"]))[:min(k, total)]
        for rang, item in enumerate(tries, 1):
            item["rang"] = rang
            item["similarite_cosinus"] = max(item["similarites"].values())
            item["classement"] = "RRF(60) texte+image" if hybride else "cosinus CLIP"
            if numero_figure:
                item["classement"] += " + filtre figure " + numero_figure.group(1)
        return tries

    def recuperer_sources(self, question=None, type_resultat="tous", k=6, image_id=None, document_id=None):
        debut = time.perf_counter()
        sources = self.rechercher(question, image_id, type_resultat, k, document_id)
        resultat = {"sources": sources, "images": [], "duree_secondes": 0.0}
        # Les pixels sont transmis au LLM dans le contexte multimodal, pas seulement
        # les chemins. Les chemins restent relatifs et sont résolus côté service.
        for source in sources:
            if source["metadata"]["type"] == "image" and len(resultat["images"]) < 3:
                resultat["images"].append({"reference": source["reference"],
                    "base64": image_base64(self.image_source(source["id"]))})
        resultat["duree_secondes"] = time.perf_counter() - debut
        return resultat
