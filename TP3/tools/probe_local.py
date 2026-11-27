"""Vérifier vision et appels d’outils avant de sélectionner le modèle du TP."""
import json
from tp3rag.config import ROOT, ecrire_json
from tp3rag.llm import verifier_capacites

resultat = verifier_capacites()
ecrire_json(ROOT / ".cache" / "capabilities.json", resultat)
print(json.dumps(resultat, indent=2, ensure_ascii=False), flush=True)
if not (resultat["vision"]["succes"] and resultat["tool_calling"]["succes"]):
    raise SystemExit("Le modèle local n’a pas passé les deux contrôles réels.")
