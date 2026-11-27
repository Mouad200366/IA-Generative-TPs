"""Commandes reproductibles : préparation, indexation, recherche et validation."""
import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from .config import ROOT, Settings


def preparer(pdf=None, download=False):
    settings = Settings()
    cible = settings.data / "pdfs" / "attention_is_all_you_need.pdf"
    cible.parent.mkdir(exist_ok=True)
    if pdf:
        source = Path(pdf)
        if not source.is_file():
            raise FileNotFoundError("Le PDF fourni n’existe pas.")
        if source.resolve() != cible.resolve():
            shutil.copyfile(source, cible)
    elif not cible.exists() and download:
        # Version fixée : les résultats BLEU de l’article varient entre révisions.
        requete = urllib.request.Request("https://arxiv.org/pdf/1706.03762v1", headers={"User-Agent": "TP3-Academic-RAG/1.0"})
        with urllib.request.urlopen(requete, timeout=90) as reponse:
            data = reponse.read(30 * 1024 * 1024)
        if not data.startswith(b"%PDF-"):
            raise ValueError("Le téléchargement n’est pas un PDF valide.")
        cible.write_bytes(data)
    elif not cible.exists():
        raise FileNotFoundError("Utiliser prepare --pdf chemin/article.pdf ou prepare --download.")
    return cible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="commande", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--pdf")
    p.add_argument("--download", action="store_true")
    p = sub.add_parser("index")
    p.add_argument("pdf", nargs="?")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("search")
    p.add_argument("question")
    p.add_argument("--type", choices=["texte", "image", "tous"], default="tous")
    p.add_argument("--k", type=int, default=6)
    sub.add_parser("count")
    sub.add_parser("probe")
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.commande == "prepare":
        preparer(args.pdf, args.download)
        print("PDF préparé dans le stockage local du TP3.")
        return
    if args.commande == "probe":
        from .llm import verifier_capacites
        from .config import ecrire_json
        resultat = verifier_capacites()
        ecrire_json(ROOT / ".cache" / "capabilities.json", resultat)
    else:
        from .retrieval import RagStore
        store = RagStore()
        if args.commande == "index":
            resultat = store.indexer(Path(args.pdf) if args.pdf else preparer(), force=args.force)
        elif args.commande == "search":
            resultat = store.rechercher(args.question, type_resultat=args.type, k=args.k)
        elif args.commande == "count":
            resultat = {"count": store.collection.count()}
        else:
            import asyncio
            from .validation import valider
            resultat = asyncio.run(valider(store))
    print(json.dumps(resultat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
