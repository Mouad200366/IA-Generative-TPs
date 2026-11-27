"""Tests unitaires : les vecteurs synthétiques contrôlent le stockage, pas la qualité du LLM."""
import asyncio
import hashlib
import io
from pathlib import Path

import numpy as np
import open_clip
import pymupdf
import pytest
from PIL import Image
from langchain_core.messages import AIMessage, ToolMessage

from tp3rag.agent import DocumentAgent, observations_compactes
from tp3rag.config import Settings
from tp3rag.embeddings import ClipEncoder, normaliser
from tp3rag.extraction import extraire
from tp3rag.retrieval import RagStore, reference


@pytest.fixture
def store(tmp_path):
    return RagStore(Settings(data=tmp_path / "data"))


def test_normalisation():
    x = normaliser(np.random.default_rng(42).normal(size=(8, 512)))
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-6)
    for invalide in [np.zeros((1, 512)), np.ones((1, 5)), np.full((1, 512), np.nan)]:
        with pytest.raises(ValueError):
            normaliser(invalide)


def test_tokens_exact_et_chevauchement():
    encoder = ClipEncoder.__new__(ClipEncoder)
    encoder.tokenizer = open_clip.get_tokenizer("ViT-B-32")
    texte = "The Transformer uses self attention and positional encodings. " * 30
    tous = encoder.tokenizer.encode(texte)
    chunks = list(encoder.decouper_texte(texte))
    assert all(0 < len(c["tokens"]) <= 60 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert a["tokens"][-10:] == b["tokens"][:10]
    reconstitue = chunks[0]["tokens"] + [t for c in chunks[1:] for t in c["tokens"][10:]]
    assert reconstitue == tous
    with pytest.raises(ValueError):
        list(encoder.decouper_texte(texte, chevauchement=60))


def test_pas_de_troncature():
    encoder = ClipEncoder.__new__(ClipEncoder)
    encoder.context_length = 77
    with pytest.raises(ValueError, match="trop long"):
        encoder.encoder_tokens([[1] * 76])


def test_passage_preserve_les_nombres():
    encoder = ClipEncoder.__new__(ClipEncoder)
    encoder.tokenizer = open_clip.get_tokenizer("ViT-B-32")
    chunk = next(encoder.decouper_texte("d_model=512 ; d_ff=2048 ; BLEU 28.4 et 41.0."))
    assert all(n in chunk["texte"] for n in ["512", "2048", "28.4", "41.0"])


@pytest.mark.parametrize("path", ["../secrets.json", "../../x", "C:/Windows/win.ini"])
def test_chemins_interdits(tmp_path, path):
    with pytest.raises(ValueError):
        Settings(data=tmp_path).chemin(path)


@pytest.mark.parametrize("kwargs", [{}, {"texte": ""}, {"texte": "a", "k": True},
    {"texte": "a", "k": 0}, {"texte": "a", "k": 13}, {"texte": "a", "type_resultat": "audio"},
    {"image_id": "../../secret.png"}])
def test_entrees_invalides(store, kwargs):
    with pytest.raises(ValueError):
        store.rechercher(**kwargs)


def test_collection_vide(store):
    assert store.rechercher("Transformer") == []


def test_image_enregistree(store):
    buffer = io.BytesIO()
    Image.new("RGB", (40, 20), "red").save(buffer, format="PNG")
    a = store.register_image(buffer.getvalue())
    b = store.register_image(buffer.getvalue())
    assert a == b and store.query_image(a).is_file()
    with pytest.raises(ValueError):
        store.query_image("a" * 64)


def test_persistance_et_limite_filtre(store):
    vector = [1.0] + [0.0] * 511
    store.collection.upsert(ids=["texte1"], embeddings=[vector], documents=["preuve"],
        metadatas=[{"type": "texte", "page": 1}])
    autre = RagStore(store.settings)
    assert autre.collection.count() == 1
    assert autre.rechercher("image", type_resultat="image", k=12) == []
    assert reference("texte1") == reference("texte1") != reference("texte2")


def test_extraction_et_cache(tmp_path):
    pdf = tmp_path / "exemple.pdf"
    with pymupdf.open() as d:
        p = d.new_page()
        p.insert_text((50, 50), "Texte de test reproductible")
        d.save(pdf)
    settings = Settings(data=tmp_path / "data")
    a = extraire(pdf, settings)
    b = extraire(pdf, settings)
    assert a == b and a["pages"] == 1
    assert a["document_id"] == hashlib.sha256(pdf.read_bytes()).hexdigest()
    assert a["textes"][0]["page"] == 1
    assert "Texte de test" in settings.chemin(a["textes"][0]["source"]).read_text()


def test_figure_vectorielle_reelle(tmp_path):
    pdf = tmp_path / "vectoriel.pdf"
    with pymupdf.open() as d:
        page = d.new_page()
        page.draw_rect(pymupdf.Rect(50, 60, 180, 130), color=(0, 0, 0), fill=(0.5, 0.8, 1))
        page.insert_text((60, 95), "Attention")
        page.insert_text((50, 160), "Figure 1: Diagramme vectoriel")
        pdf.write_bytes(d.tobytes())
    settings = Settings(data=tmp_path / "data")
    r = extraire(pdf, settings)
    assert r["images_integrees_uniques"] == 0
    assert len(r["images"]) == 1 and r["images"][0]["methode"] == "rendu_region"
    with Image.open(settings.chemin(r["images"][0]["source"])) as im:
        assert im.width > 100 and im.height > 100


def test_reindexation_et_remplacement(store, tmp_path, monkeypatch):
    class EncodeurUnitaire:
        """Vecteurs synthétiques : test de gestion de versions, pas de pertinence."""
        def __init__(self):
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        decouper_texte = ClipEncoder.decouper_texte
        def encoder_tokens(self, lot):
            return np.array([[1.] + [0.] * 511 for _ in lot], dtype=np.float32)
    encoder = EncodeurUnitaire()
    monkeypatch.setattr(RagStore, "encoder", property(lambda self: encoder))
    p = tmp_path / "rapport.pdf"
    def ecrire(texte):
        with pymupdf.open() as d:
            d.new_page().insert_text((50, 50), texte)
            p.write_bytes(d.tobytes())
    ecrire("Premiere version du document")
    a = store.indexer(p)
    ids = store.collection.get()["ids"]
    b = store.indexer(p, force=True)
    assert store.collection.get()["ids"] == ids and a["total"] == b["total"]
    ecrire("Nouvelle version modifiee du document")
    c = store.indexer(p)
    assert c["document_id"] != a["document_id"]
    assert store.collection.get(where={"document_id": a["document_id"]})["ids"] == []
    assert store.collection.count() == c["total"]


def test_budget_strict(monkeypatch):
    class ModeleControle:
        """Modèle factice réservé au test unitaire du compteur, sans validation LLM."""
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            if any(getattr(m, "type", "") == "tool" for m in messages):
                return AIMessage(content="Preuves insuffisantes.")
            return AIMessage(content="", tool_calls=[{"name": "consulter_pdf", "args": {"question": "attention"},
                "id": f"controle-{i}", "type": "tool_call"} for i in range(5)])

    monkeypatch.setattr("tp3rag.agent.modele_local", lambda settings: ModeleControle())
    effectues = []
    def retrieval(**kwargs):
        effectues.append(kwargs)
        return {"sources": [], "images": []}
    resultat = DocumentAgent(retrieval).ask("Tester le compteur.")
    assert len(effectues) == resultat["nombre_appels"] == 3
    assert resultat["budget_respecte"]


@pytest.mark.parametrize('question,image_seule', [('', True), ('Décris cette architecture.', False)])
def test_entree_image_seule_ou_hybride(store, monkeypatch, question, image_seule):
    """Contrôle unitaire du routage ; le modèle factice n'évalue pas le LLM."""
    class PlanificateurControle:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            if any(getattr(m, 'type', '') == 'tool' for m in messages):
                return AIMessage(content='Preuves insuffisantes.')
            return AIMessage(content='', tool_calls=[{'name': 'consulter_pdf',
                'args': {'question': 'architecture'}, 'id': 'image-controle', 'type': 'tool_call'}])

    monkeypatch.setattr('tp3rag.agent.modele_local', lambda settings: PlanificateurControle())
    buffer = io.BytesIO()
    Image.new('RGB', (30, 20), 'blue').save(buffer, format='PNG')
    ident = store.register_image(buffer.getvalue())
    appels = []
    def retrieval(**kwargs):
        appels.append(kwargs)
        return {'sources': [], 'images': []}
    resultat = DocumentAgent(retrieval, store.settings).ask(question, image_id=ident)
    assert resultat['nombre_appels'] == 1
    assert appels[0]['image_id'] == ident
    assert (appels[0]['question'] is None) == image_seule


def test_relecture_peut_declencher_une_recherche(monkeypatch):
    """Contrôle du graphe avec planificateur factice, sans mesure de qualité LLM."""
    class PlanificateurControle:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            outils = [m for m in messages if getattr(m, 'type', '') == 'tool']
            verification = isinstance(messages[-1].content, str) and 'Vérifie la couverture' in messages[-1].content
            if not outils or verification:
                return AIMessage(content='', tool_calls=[{'name': 'consulter_pdf',
                    'args': {'question': 'technical definition' if verification else 'initial evidence'},
                    'id': 'controle-' + str(len(outils)), 'type': 'tool_call'}])
            return AIMessage(content='Preuves insuffisantes.')

    monkeypatch.setattr('tp3rag.agent.modele_local', lambda settings: PlanificateurControle())
    appels = []
    def retrieval(**kwargs):
        appels.append(kwargs)
        return {'sources': [], 'images': []}
    resultat = DocumentAgent(retrieval).ask('Expliquer un mécanisme documentaire.')
    assert resultat['nombre_appels'] == len(appels) == 2
    assert resultat['controle_generation']['relecture_couverture_effectuee']
    assert appels[0]['question'] != appels[1]['question']
    assert resultat['budget_respecte']


def test_contexte_deduplique_preserve_traces_et_erreurs():
    import json
    anciens = ToolMessage(content=json.dumps({'sources': [{'reference': 'S1111', 'contenu': 'preuve A'},
        {'reference': 'S2222', 'contenu': 'preuve B'}]}), tool_call_id='appel-1')
    recents = ToolMessage(content=json.dumps({'sources': [{'reference': 'S2222', 'contenu': 'preuve B'},
        {'reference': 'S3333', 'contenu': 'preuve C'}]}), tool_call_id='appel-2')
    erreur = ToolMessage(content=json.dumps({'erreur': 'ValueError'}), tool_call_id='appel-3')
    messages = [anciens, recents, erreur]
    originaux = [m.content for m in messages]
    compact = observations_compactes(messages, limite=2)
    observation = json.loads(compact[-1].content)
    assert [s['reference'] for s in observation['sources']] == ['S2222', 'S3333']
    assert observation['observations_sans_preuves'] == [{'erreur': 'ValueError'}]
    assert [m.content for m in messages] == originaux
    assert [m.tool_call_id for m in compact] == ['appel-1', 'appel-2', 'appel-3']
