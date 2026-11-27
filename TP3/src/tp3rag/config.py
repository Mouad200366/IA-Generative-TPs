"""Configuration locale commune au notebook, à Streamlit et à MCP."""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING = {"model": "ViT-B-32", "pretrained": "laion2b_s34b_b79k",
             "dimension": 512, "chunk_tokens": 60, "overlap": 10, "metric": "cosine"}
FINGERPRINT = hashlib.sha256(json.dumps(EMBEDDING, sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Settings:
    data: Path = field(default_factory=lambda: Path(os.environ.get("TP3_DATA_DIR", ROOT / "data")))
    model: str = field(default_factory=lambda: os.environ.get("TP3_LLM_MODEL", "qwen3-vl:4b-instruct-q4_K_M"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("TP3_OLLAMA_URL", "http://127.0.0.1:11434"))
    clip_device: str = field(default_factory=lambda: os.environ.get("TP3_CLIP_DEVICE", "auto"))

    def __post_init__(self):
        self.data = Path(self.data).resolve()
        if urlparse(self.ollama_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Ce TP utilise uniquement un serveur Ollama local.")
        self.data.mkdir(parents=True, exist_ok=True)

    def chemin(self, relatif: str) -> Path:
        """Résoudre uniquement un fichier du stockage autorisé, y compris via un lien."""
        p = (self.data / relatif).resolve()
        if not p.is_relative_to(self.data):
            raise ValueError("Chemin extérieur au stockage documentaire autorisé.")
        return p


def lire_json(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def ecrire_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    temporaire = p.with_suffix(p.suffix + ".tmp")
    temporaire.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporaire.replace(p)
