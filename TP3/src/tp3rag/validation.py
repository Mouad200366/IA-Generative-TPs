"""Validation réelle sur 15 cas annotés ; aucun LLM fictif dans ce protocole."""
import asyncio
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .agent import INSTRUCTIONS, agent_direct
from .config import EMBEDDING, ROOT, ecrire_json, lire_json
from .extraction import extraire


async def valider(store, prioritaires=()):
    debut = time.perf_counter()
    settings = store.settings
    if not lire_json(ROOT / ".cache" / "capabilities.json"):
        from .llm import verifier_capacites
        ecrire_json(ROOT / ".cache" / "capabilities.json", verifier_capacites(settings))
    annotation = lire_json(ROOT / "tests" / "cases.json")
    ids = {c['id'] for c in annotation['cases']}
    if not set(prioritaires) <= ids:
        raise ValueError("Scénario prioritaire inconnu.")
    cas_ordonnes = sorted(annotation['cases'], key=lambda c: c['id'] not in prioritaires)
    pdf = settings.data / "pdfs" / "attention_is_all_you_need.pdf"
    extraction = extraire(pdf, settings)
    if extraction["document_id"] != annotation["document_sha256"]:
        raise ValueError("Ce corpus diffère du PDF annoté : vérifier les pages et chiffres avant de mesurer Hit@k.")
    avant = set(store.collection.get()["ids"])
    def empreinte_vecteurs():
        donnees = store.collection.get(include=['embeddings'])
        digest = hashlib.sha256()
        for i in sorted(range(len(donnees['ids'])), key=lambda i: donnees['ids'][i]):
            digest.update(donnees['ids'][i].encode())
            digest.update(np.asarray(donnees['embeddings'][i], dtype='<f4').tobytes())
        return digest.hexdigest()
    empreinte_avant = empreinte_vecteurs()
    # Réutiliser le corpus inchangé sans le réécrire : même des upserts des
    # vecteurs existants peuvent modifier leur arrondi après normalisation Chroma.
    reindex = store.indexer(pdf)
    apres = set(store.collection.get()["ids"])
    assert avant == apres
    # La seconde ouverture s’effectue dans un autre processus Python.
    persiste = subprocess.run([sys.executable, "-m", "tp3rag.cli", "count"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=60, check=True)
    nombre_persistant = json.loads(persiste.stdout)["count"]
    brut = store.collection.get(include=["metadatas", "embeddings"])
    normes = np.linalg.norm(np.asarray(brut["embeddings"]), axis=1)
    assert np.isfinite(brut["embeddings"]).all() and np.allclose(normes, 1, atol=1e-5)
    assert all(m["page"] >= 1 and m["document_id"] == extraction["document_id"]
               and not m["source"].startswith(("/", "C:")) for m in brut["metadatas"])
    assert all(m.get("tokens", 0) <= 60 for m in brut["metadatas"])
    figures = {x["figure"]: x for x in extraction["images"] if x["figure"]}
    image_ids = {n: store.register_image(settings.chemin(im["source"]).read_bytes()) for n, im in figures.items()}
    resultat = {"date_utc": datetime.now(timezone.utc).isoformat(), "statut": "en_cours",
        "corpus": {"pdf": extraction["pdf"], "document_sha256": extraction["document_id"],
                   "pages": extraction["pages"], "chunks": reindex["chunks"], "images": reindex["images"],
                   "images_integrees_uniques": extraction["images_integrees_uniques"],
                   "bleu_anglais_francais_dans_ce_pdf": 41.0},
        "embedding": EMBEDDING, "modele_local": settings.model,
        "instructions_agent_sha256": hashlib.sha256(INSTRUCTIONS.encode()).hexdigest(),
        "capacites_llm": lire_json(ROOT / ".cache" / "capabilities.json"),
        "infrastructure": {"reindexation_sans_doublon": avant == apres,
            "persistance_autre_processus": nombre_persistant == len(apres),
            "norme_min": float(normes.min()), "norme_max": float(normes.max()),
            "dimension": 512, "vecteurs_finis": True, "metadonnees_valides": True},
        "protocole": {"nombre_cas": len(annotation["cases"]), "annotation": annotation["annotation"],
            "retrieval": "Hit@6 par page pour le texte ; Hit@1 pour les figures. Les pages annotées ne sont pas des jugements exhaustifs de chaque chunk.",
            "reponses": "Contrôles de motifs, d’appels et de références : ils ne constituent pas une mesure complète de vérité ou de fidélité sémantique.",
            "image_identique": "Test de fonctionnement uniquement, sans mesure de généralisation.",
            "hybride": "Fusion RRF, constante 60, au plus 48 candidats par entrée. Les cosinus texte/image restent séparés."},
        "cas": []}
    resultat['protocole']['ordre_execution'] = [c['id'] for c in cas_ordonnes]
    resultat['protocole']['cas_prioritaires'] = list(prioritaires)
    resultat['protocole']['criteres_lexicaux_inchanges'] = True
    agent = agent_direct(store, extraction["document_id"])
    for index, case in enumerate(cas_ordonnes, 1):
        print(f"Validation réelle {index}/15 : {case['id']}", flush=True)
        image_id = image_ids.get(case.get("image_figure"))
        cas = {"id": case["id"], "question": case["question"], "pages_annotees": case["pages"], "statut": "succes"}
        try:
            if not case.get("salutation"):
                k = 1 if case["type"] == "image" else 6
                r = store.rechercher(case["query"], image_id, case["type"], k, extraction["document_id"])
                cas["recherche"] = {"k": k, "resultats": r, "pages": [s["metadata"]["page"] for s in r]}
                if case["pages"]:
                    cas["recherche"]["hit"] = any(s["metadata"]["page"] in case["pages"] for s in r)
                if case.get("identite"):
                    cas["recherche"]["similarite_identique"] = r[0]["similarite_cosinus"]
            sortie = await agent.aask(case["question"], image_id, case["type"], 8 if case["id"] == "base" else 6)
            cas["agent"] = sortie
            cas["controles"] = {"motifs_reponse": all(re.search(p, sortie["reponse"], re.I) for p in case["patterns"]),
                "references_connues": not sortie["references_inconnues"],
                "citations_presentes": bool(sortie["sources_citees"]),
                "budget_respecte": sortie["budget_respecte"],
                "outil_si_necessaire": sortie["nombre_appels"] == 0 if case.get("salutation") else sortie["nombre_appels"] > 0,
                "image_utilisee_pour_recherche": any(c["arguments"].get("image_id") == image_id for c in sortie["appels_outils"]) if image_id else None}
        except Exception as exc:
            cas.update({"statut": "echec", "erreur": type(exc).__name__})
            print("Échec :", type(exc).__name__, flush=True)
        resultat["cas"].append(cas)
        ecrire_json(ROOT / ".cache" / "evaluation_progress.json", resultat)
    # Extension MCP : découverte, appel du véritable outil puis graphe complet.
    try:
        from .mcp_transport import ouvrir_client
        async with ouvrir_client(settings, extraction["document_id"]) as client:
            recherche = await client["rechercher"](question="scaled dot product attention", type_resultat="image", k=2, image_id=None)
            agent_mcp = await client["agent"].aask("Explique la figure 2 à partir du document.", type_resultat="image", k=2)
            resultat["mcp"] = {"statut": "succes", "outils_decouverts": client["outils"],
                "sources_recuperees": len(recherche["sources"]), "images_recues": len(recherche["images"]), "agent": agent_mcp}
    except Exception as exc:
        resultat["mcp"] = {"statut": "echec", "erreur": type(exc).__name__}
    hits = [c["recherche"]["hit"] for c in resultat["cas"] if "hit" in c.get("recherche", {})]
    executions = [c["agent"] for c in resultat["cas"] if "agent" in c]
    resultat["mesures"] = {"recherche_hit_par_page": sum(hits), "recherche_cas_annotes": len(hits),
        "hit_rate_pages": sum(hits) / len(hits) if hits else None,
        "agents_executes": len(executions), "nombre_cas": 15,
        "references_inconnues": sum(len(s["references_inconnues"]) for s in executions),
        "corrections_format_reelles": sum(s["controle_generation"]["correction_format"] for s in executions),
        "references_initiales_inconnues": sum(len(s["controle_generation"]["references_initiales_inconnues"]) for s in executions),
        "images_requete_effectivement_utilisees": all(c.get("controles", {}).get("image_utilisee_pour_recherche") is not False for c in resultat["cas"]),
        "controles_motifs_reussis": sum(c.get("controles", {}).get("motifs_reponse", False) for c in resultat["cas"]),
        "budget_3_respecte": all(s["budget_respecte"] for s in executions),
        "temps_agent_moyen_secondes": float(np.mean([s["duree_secondes"] for s in executions])) if executions else None}
    resultat["versions"] = {n: importlib.metadata.version(n) for n in
        ["torch", "torchvision", "open-clip-torch", "chromadb", "langgraph", "langchain-ollama", "streamlit", "mcp"]}
    empreinte_apres = empreinte_vecteurs()
    assert empreinte_avant == empreinte_apres, 'Les vecteurs indexés ont changé.'
    resultat['infrastructure'].update({'vecteurs_sha256_avant': empreinte_avant,
        'vecteurs_sha256_apres': empreinte_apres, 'vecteurs_existants_preserves': True})
    initial = lire_json(ROOT / 'evaluations' / 'reference_initiale' / 'resultats_tp3.json')
    if initial:
        precedents = {c['id']: c for c in initial['cas']}
        resultat['comparaison_reference'] = {
            'reference': 'evaluations/reference_initiale/resultats_tp3.json',
            'motifs_avant': initial['mesures']['controles_motifs_reussis'],
            'motifs_apres': resultat['mesures']['controles_motifs_reussis'],
            'cas': [{'id': c['id'], 'prioritaire': c['id'] in prioritaires,
                'motifs_avant': precedents[c['id']].get('controles', {}).get('motifs_reponse'),
                'motifs_apres': c.get('controles', {}).get('motifs_reponse')}
                for c in resultat['cas']]}
    resultat["duree_secondes"] = time.perf_counter() - debut
    resultat["statut"] = "execute" if all(c["statut"] == "succes" for c in resultat["cas"]) and resultat["mcp"]["statut"] == "succes" else "partiel"
    resultat["streamlit"] = lire_json(ROOT / ".cache" / "streamlit_validation.json", {"statut": "non_execute"})
    ecrire_json(ROOT / "resultats_tp3.json", resultat)
    return resultat
