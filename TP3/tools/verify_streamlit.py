"""Parcours AppTest avec les octets réels des fichiers, OpenCLIP et Ollama."""
import hashlib
import json
import time
import urllib.request

import numpy as np
from streamlit.testing.v1 import AppTest

from tp3rag.config import ROOT, Settings, ecrire_json
from tp3rag.retrieval import RagStore


def empreinte_index(store):
    donnees = store.collection.get(include=['embeddings'])
    digest = hashlib.sha256()
    for i in sorted(range(len(donnees['ids'])), key=lambda i: donnees['ids'][i]):
        digest.update(donnees['ids'][i].encode())
        digest.update(np.asarray(donnees['embeddings'][i], dtype='<f4').tobytes())
    return digest.hexdigest()


def bouton(app, label):
    return next(b for b in app.button if b.label == label)


debut = time.perf_counter()
store = RagStore(Settings())
avant = empreinte_index(store)
pdf = store.settings.data / 'pdfs' / 'attention_is_all_you_need.pdf'
figures = store.collection.get(where={'type': 'image'}, include=['metadatas'])
i = next(i for i, m in enumerate(figures['metadatas']) if m['figure'] == 1)
image = store.image_source(figures['ids'][i])
app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=180).run()
assert not app.exception and len(app.metric) == 3
assert bouton(app, 'Rechercher avec l’image seule').disabled

# Les composants reçoivent les vrais octets. Le geste du sélecteur natif et le
# transfert navigateur ne sont pas présentés comme validés par AppTest.
app.file_uploader[0].set_value((pdf.name, pdf.read_bytes(), 'application/pdf')).run()
assert app.file_uploader[0].value.getvalue() == pdf.read_bytes()
bouton(app, 'Indexer le PDF').click().run(timeout=180)
assert not app.exception and not app.error and app.success
app.selectbox[0].select(pdf.name).run()
assert app.selectbox[0].value == pdf.name and len(app.metric) == 3
app.file_uploader[1].set_value(('figure1.png', image.read_bytes(), 'image/png')).run()
assert app.file_uploader[1].value.getvalue() == image.read_bytes()
assert not bouton(app, 'Rechercher avec l’image seule').disabled

app.chat_input[0].set_value('Utilise la question et cette image pour expliquer la structure encodeur-décodeur.').run(timeout=240)
assert not app.exception and not app.error
hybride = app.session_state.messages[-1]['sortie']
assert hybride['nombre_appels'] > 0 and hybride['sources']
assert any(c['arguments']['image_id'] and c['arguments']['question'] for c in hybride['appels_outils'])
assert any(s['classement'].startswith('RRF') for s in hybride['sources'])
assert any('Sources consultées' in e.label for e in app.expander)
assert any('Appels d’outils' in e.label for e in app.expander)

bouton(app, 'Rechercher avec l’image seule').click().run(timeout=240)
assert not app.exception and not app.error
image_seule = app.session_state.messages[-1]['sortie']
assert image_seule['nombre_appels'] > 0 and image_seule['sources']
assert all(c['arguments']['question'] is None and c['arguments']['image_id'] for c in image_seule['appels_outils'])
apres = empreinte_index(store)
assert avant == apres

rapport = {'statut': 'succes',
    'methode': 'Streamlit AppTest, fichiers réels, sans mock de la recherche ou du LLM',
    'demarrage': True, 'indicateurs': len(app.metric), 'chat_reel': True,
    'import_pdf_octets_reels': True, 'indexation_et_selection': True,
    'import_image_octets_reels': True, 'recherche_hybride': hybride,
    'recherche_image_seule': image_seule,
    'sources_et_appels_affiches': True, 'vecteurs_index_preserves': avant == apres,
    'vecteurs_sha256': apres, 'duree_secondes': time.perf_counter() - debut,
    'gestes_upload_navigateur': 'non_valide',
    'note': 'AppTest valide les composants et leur backend avec les vrais fichiers. Le sélecteur natif et le transfert navigateur restent à vérifier manuellement.'}
try:
    with urllib.request.urlopen('http://127.0.0.1:8503/_stcore/health', timeout=3) as reponse:
        rapport['serveur_http'] = {'port': 8503, 'status': reponse.status, 'health': reponse.read().decode()}
except OSError:
    rapport['serveur_http'] = {'statut': 'non_demarre_ou_non_accessible'}
ecrire_json(ROOT / '.cache' / 'streamlit_validation.json', rapport)
print(json.dumps({k: v for k, v in rapport.items() if not k.startswith('recherche_')}, indent=2, ensure_ascii=False))
