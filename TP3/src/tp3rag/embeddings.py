"""OpenCLIP officiel, tokenisation exacte et inférence par lots (support 18–30)."""
import functools
import logging

import numpy as np
import open_clip
import torch
from PIL import Image

from .config import EMBEDDING, Settings


def normaliser(vecteurs):
    x = np.asarray(vecteurs, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 512 or not np.isfinite(x).all():
        raise ValueError("Embeddings invalides : matrice finie de dimension 512 attendue.")
    normes = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(normes < 1e-8):
        raise ValueError("Un embedding nul ne peut pas être normalisé.")
    return x / normes


class ClipEncoder:
    def __init__(self, settings: Settings):
        torch.set_num_threads(4)
        device = settings.clip_device
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            EMBEDDING["model"], pretrained=EMBEDDING["pretrained"],
            cache_dir=str(settings.data / "models"))
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(EMBEDDING["model"])
        self.context_length = self.model.context_length

    def decouper_texte(self, texte, taille=60, chevauchement=10):
        if type(taille) is not int or type(chevauchement) is not int or not 0 <= chevauchement < taille <= 75:
            raise ValueError("Taille/chevauchement de tokens invalides.")
        # Conserver les plages de caractères du texte nettoyé. decode() de CLIP
        # transforme par exemple 512 en « 5 1 2 » et 28.4 en « 2 8 . 4 » :
        # ce décodage convient mal aux preuves numériques d’un RAG.
        propre = self.tokenizer.clean_fn(texte)
        tokens, plages = [], []
        for match in self.tokenizer.pat.finditer(propre):
            mot = match.group()
            octets = mot.encode("utf-8")
            encode = "".join(self.tokenizer.byte_encoder[b] for b in octets)
            position = 0
            for morceau in self.tokenizer.bpe(encode).split(" "):
                longueur = len(morceau.replace("</w>", ""))
                gauche = len(octets[:position].decode("utf-8", errors="ignore"))
                droite = len(octets[:position + longueur].decode("utf-8", errors="replace"))
                tokens.append(self.tokenizer.encoder[morceau])
                plages.append((match.start() + gauche, match.start() + droite))
                position += longueur
        assert tokens == self.tokenizer.encode(texte)
        for debut in range(0, len(tokens), taille - chevauchement):
            ids = tokens[debut:debut + taille]
            if ids:
                fin = min(debut + taille, len(tokens))
                yield {"texte": propre[plages[debut][0]:plages[fin - 1][1]].strip(),
                       "tokens": ids, "debut_token": debut}
            if debut + taille >= len(tokens):
                break

    @torch.inference_mode()
    def encoder_tokens(self, listes):
        # Ne pas décoder puis retokeniser les chunks : conserver exactement leurs
        # 60 tokens et leur chevauchement, sans troncature implicite du tokenizer.
        matrice = torch.zeros((len(listes), self.context_length), dtype=torch.long)
        for i, ids in enumerate(listes):
            if len(ids) > self.context_length - 2:
                raise ValueError("Texte trop long pour CLIP : découper la requête (75 tokens hors marqueurs).")
            sequence = [self.tokenizer.sot_token_id, *ids, self.tokenizer.eot_token_id]
            matrice[i, :len(sequence)] = torch.tensor(sequence)
        return normaliser(self.model.encode_text(matrice.to(self.device)).float().cpu().numpy())

    def encoder_textes(self, textes):
        if not textes or any(not isinstance(t, str) or not t.strip() for t in textes):
            raise ValueError("Un lot de textes non vides est requis.")
        return self.encoder_tokens([self.tokenizer.encode(t) for t in textes])

    @torch.inference_mode()
    def encoder_images(self, chemins):
        tenseurs = []
        for chemin in chemins:
            with Image.open(chemin) as im:
                tenseurs.append(self.preprocess(im.convert("RGB")))
        return normaliser(self.model.encode_image(torch.stack(tenseurs).to(self.device)).float().cpu().numpy())


@functools.lru_cache(maxsize=2)
def get_encoder(data: str, device: str = "auto"):
    """Une instance par répertoire de modèles/périphérique et par processus."""
    return ClipEncoder(Settings(data=data, clip_device=device))
