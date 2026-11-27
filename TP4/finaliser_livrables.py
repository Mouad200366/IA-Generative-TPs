"""Finaliser les documents et vérifier l'archive, sans entraîner ni télécharger."""
import ast
import copy
import hashlib
import json
import re
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / 'TP4_Architecture_Transformer.ipynb'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def lire(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ecrire(path, objet):
    path.write_text(json.dumps(objet, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8', newline='\n')


def git(*args, entree=None):
    # Entrée binaire : empêcher la conversion automatique LF -> CRLF sous Windows.
    sortie = subprocess.check_output(['git', *args], cwd=ROOT.parent,
        input=entree.encode('utf-8') if entree is not None else None)
    return sortie.decode('utf-8').splitlines()


def fichiers_publics():
    candidats = set(git('ls-files', '--', 'TP4'))
    candidats.update(git('ls-files', '--others', '--exclude-standard', '--', 'TP4'))
    return sorted(ROOT.parent / p for p in candidats)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    r = lire(ROOT / 'resultats_tp4.json')
    preuve = lire(ROOT / 'verification_execution.json')
    initial_path = cache / 'resultats_avant_finalisation.json'
    if initial_path.exists():
        initial = lire(initial_path)
        comparable = copy.deepcopy(r)
        # La publication retire uniquement la licence non autorisée des corpus.
        for rapport in (initial, comparable):
            rapport.get('corpus', {}).pop('licence', None)
        assert all(comparable[k] == initial[k] for k in initial if k not in {'date_utc', 'tests_automatises'})
    assert r['statut'] == 'termine' and preuve['statut'] == 'succes'
    nb = nbformat.read(NOTEBOOK, as_version=4)
    codes = [c for c in nb.cells if c.cell_type == 'code']
    assert len(codes) == 8
    assert [c.execution_count for c in codes] == list(range(1, 9))
    assert not any(o.output_type == 'error' for c in codes for o in c.outputs)
    avant = copy.deepcopy(codes)
    hash_avant = sha(NOTEBOOK)

    # Commentaires dérivés des mesures ; aucune modification des cellules Python.
    marqueur = '\n\n<!-- lecture-des-resultats -->\n'
    sinus = r['sinus']['modeles']
    meilleur = min(sinus, key=lambda n: sinus[n]['metriques_test']['mse'])
    notes = {
        4: f"Dans cette exécution, **{meilleur}** donne la plus faible MSE de test "
           f"({sinus[meilleur]['metriques_test']['mse']:.8g}). Le Transformer obtient "
           "une erreur intermédiaire avec davantage de paramètres et un temps plus long. "
           "Ces observations concernent uniquement cette sinusoïde et cette graine.",
        8: "Les trois méthodes sont comparées sur **4 932 cibles identiques**. "
           "MAE et RMSE sont exprimées en kW ; la MSE est en kW². "
           "Le modèle retenu à l'époque 17 améliore les deux références sur MAE et RMSE. "
           "Les sept jours affichés comportent ici **0 heure sans prédiction** ; "
           "le tracé conserverait les interruptions si des fenêtres manquaient.",
        12: f"La perte de validation retenue est "
            f"**{r['mini_gpt']['meilleure_perte_suivie']:.6f} par caractère**, "
            f"soit une perplexité de **{r['mini_gpt']['perplexite_validation_par_caractere']:.6f}**. "
            "Les sorties enregistrées contiennent des fragments familiers mais aussi des mots "
            "déformés et des phrases incorrectes, particulièrement à température élevée. "
            "Ce résultat sur un petit corpus à patrons récurrents ne mesure pas une compréhension. "
            "La validation sert au choix des poids ; aucun test linguistique indépendant n'est utilisé.",
        14: f"L'exactitude est de **{100*r['mini_bert']['accuracy_validation_selection']:.2f} %** "
            "sur les **144 positions sélectionnées** de validation, y compris les positions "
            "sélectionnées inchangées. Sur les seules positions remplacées par MASK, elle vaut "
            f"**{100*r['mini_bert']['accuracy_validation_remplacements_mask']:.2f} %**. "
            "Les 38 UNK sur 954 mots de validation (3,98 %) ne sont pas sélectionnés pour le MLM. "
            "Les patrons récurrents, le petit corpus et le choix des poids sur cette même "
            "validation limitent la portée du score ; il ne s'agit pas d'un test indépendant.",
    }
    for index, texte in notes.items():
        assert nb.cells[index].cell_type == 'markdown'
        nb.cells[index].source = nb.cells[index].source.split(marqueur)[0] + marqueur + texte
    assert avant == [c for c in nb.cells if c.cell_type == 'code']
    nbformat.validate(nb)
    NOTEBOOK.write_text(nbformat.writes(nb), encoding='utf-8', newline='\n')

    modeles = [*sinus.values(), r['energie'], r['mini_gpt'], r['mini_bert']]
    for modele in modeles:
        assert sha(ROOT / modele['poids']) == modele['poids_sha256']
        assert modele['configuration']['architecture_sha256'] == sha(ROOT / 'modeles.py')
        relu = modele.get('rechargement', modele.get('tests', {}).get('rechargement'))
        assert relu['reussi'] and relu['identique'] and relu['ecart_max'] <= 1e-6
    for nom, valeur in r['corpus']['sha256'].items():
        assert sha(ROOT / nom) == valeur
    assert all(len(g['texte_nouveau']) == g['nouveaux_caracteres'] == 200
               for g in r['mini_gpt']['generations'])

    # Ces tests courts contrôlent aussi les rapports finaux, sans entraîner les modèles.
    debut = time.perf_counter()
    test = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
                          capture_output=True, text=True, encoding='utf-8', errors='replace')
    log = test.stdout + test.stderr
    (cache / 'pytest_livrables.log').write_text(log, encoding='utf-8')
    print(log.strip())
    assert test.returncode == 0, 'Tests en échec : consulter .cache/pytest_livrables.log'
    nombre = re.search(r'(\d+) passed', log)
    assert nombre
    tests = {'statut': 'succes', 'commande': 'python -m pytest -q',
             'returncode': test.returncode, 'reussis': int(nombre.group(1)),
             'secondes_processus': time.perf_counter() - debut,
             'date_utc': datetime.now(timezone.utc).isoformat(),
             'log_sha256': sha(cache / 'pytest_livrables.log')}
    r['tests_automatises'] = tests
    ecrire(ROOT / 'resultats_tp4.json', r)

    fichiers = fichiers_publics()
    interdits = {'.cache', '.venv', 'data', 'supports', '__pycache__', '.pytest_cache'}
    for p in fichiers:
        rel = p.relative_to(ROOT)
        assert not interdits.intersection(rel.parts), rel
        assert p.suffix.lower() not in {'.pdf', '.safetensors', '.zip', '.pem', '.key'}, rel
        assert p.stat().st_size < 5_000_000, rel
        if p.suffix in {'.py', '.json', '.ipynb', '.md', '.toml', '.txt', '.lock'}:
            contenu = p.read_text(encoding='utf-8')
            assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]', contenu), rel
            assert not re.search(r'(?:hf_|ghp_)[A-Za-z0-9]{25,}', contenu), rel
            assert not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----', contenu), rel
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=str(rel))
    # Chaque image intégrée est strictement l'une des huit figures produites ici.
    figures = {p.read_bytes() for p in (ROOT / 'figures').glob('*.png')}
    import base64
    images = [base64.b64decode(o['data']['image/png']) for c in codes
              for o in c.outputs if 'image/png' in o.get('data', {})]
    assert len(images) == len(figures) == 8 and set(images) == figures
    assert all(c.get('attachments', {}) == {} for c in nb.cells)
    assert not re.search(r'<img|!\[', '\n'.join(c.source for c in nb.cells if c.cell_type == 'markdown'))
    for c in codes:
        ast.parse(c.source)
    assert (ROOT / '.python-version').read_text().strip() == '3.11'
    modifies = git('diff', '--name-only', 'HEAD')
    nouveaux = git('ls-files', '--others', '--exclude-standard')
    assert all(p.startswith('TP4/') or p == 'README.md' for p in modifies + nouveaux)
    subprocess.run(['git', 'diff', '--check'], cwd=ROOT.parent, check=True, capture_output=True)
    chemins = ['TP4/' + p for p in ['data/test.txt', '.venv/test', 'supports/test.ipynb',
               '.cache/test', 'test.pdf', '.env', 'secrets.json', '__pycache__/test.pyc']]
    ignores = git('check-ignore', '--stdin', entree='\n'.join(chemins) + '\n')
    assert set(ignores) == set(chemins)

    preuve.update(notebook_sha256=sha(NOTEBOOK), tests_automatises=tests,
        finalisation={'date_utc': datetime.now(timezone.utc).isoformat(),
            'notebook_avant_commentaires_sha256': hash_avant,
            'modification_apres_execution': 'Markdown uniquement : interprétation des résultats',
            'code_compteurs_sorties_inchanges': avant == [c for c in nb.cells if c.cell_type == 'code'],
            'figures_originales_verifiees': len(images), 'python_syntaxiquement_valide': True,
            'six_poids_empreintes_et_rechargements_valides': True,
            'resultats_scientifiques_inchanges_depuis_premiere_execution': True if initial_path.exists() else None,
            'code_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in fichiers if p.suffix == '.py'},
            'modifications_limitees_tp4_et_readme_principal': True, 'exclusions_git_verifiees': ignores,
            'aucun_chemin_personnel_ou_secret_detecte': True,
            'commit_effectue': False, 'push_effectue': False})
    ecrire(ROOT / 'verification_execution.json', preuve)
    fichiers.append(ROOT / 'verification_execution.json')
    manifeste = {p.relative_to(ROOT).as_posix(): {'sha256': sha(p), 'octets': p.stat().st_size}
                 for p in fichiers}
    archive = cache / 'TP4_verification_ChatGPT.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in fichiers:
            z.write(p, 'TP4/' + p.relative_to(ROOT).as_posix())
        z.writestr('TP4/MANIFESTE_SHA256.json', json.dumps(manifeste, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for nom, info in manifeste.items():
            assert hashlib.sha256(z.read('TP4/' + nom)).hexdigest() == info['sha256']
        assert len(z.namelist()) == len(manifeste) + 1
    bilan = {'archive': archive.relative_to(ROOT).as_posix(), 'sha256': sha(archive),
             'octets': archive.stat().st_size, 'fichiers': len(manifeste) + 1,
             'notebook_sha256': sha(NOTEBOOK), 'resultats_sha256': sha(ROOT / 'resultats_tp4.json')}
    ecrire(cache / 'archive_verification.json', bilan)
    print(json.dumps(bilan, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
