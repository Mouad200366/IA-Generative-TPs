"""Retirer les reproductions des artefacts publics, sans exécuter les expériences."""
import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
MASQUE = '[Contenu documentaire retiré de la version publique ; disponible dans la copie locale complète.]'
NOTE = 'Version publique dérivée de l’exécution locale : textes récupérés, réponses documentaires et figures reproduites retirés ; métriques, identifiants et pages conservés. Aucune expérience relancée.'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lire(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ecrire(path, valeur):
    path.write_text(json.dumps(valeur, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def masquer(valeur, parent=''):
    if isinstance(valeur, list):
        return [masquer(v, parent) for v in valeur]
    if not isinstance(valeur, dict):
        return valeur
    resultat = {}
    for cle, contenu in valeur.items():
        if cle in {'contenu', 'reponse', 'base64', 'image_url'} and isinstance(contenu, str):
            resultat[cle] = MASQUE
        elif parent == 'arguments' and cle == 'question' and isinstance(contenu, str):
            resultat[cle] = MASQUE
        else:
            resultat[cle] = masquer(contenu, cle)
    return resultat


def verifier_structure(avant, apres):
    """Les métriques et la provenance structurée ne doivent pas être modifiées."""
    if isinstance(avant, dict):
        for k, v in avant.items():
            assert k in apres
            if k in {'id', 'reference', 'page', 'document_id', 'document_sha256'}:
                assert v == apres[k]
            verifier_structure(v, apres[k])
    elif isinstance(avant, list):
        assert len(avant) == len(apres)
        for a, b in zip(avant, apres):
            verifier_structure(a, b)
    elif isinstance(avant, (int, float, bool)) or avant is None:
        assert avant == apres


def sauvegarder():
    dossier = ROOT / '.cache' / 'publication_complete'
    if dossier.exists():
        courant = lire(ROOT / 'resultats_tp3.json')
        if courant == masquer(courant):
            return dossier
        dossier = ROOT / '.cache' / ('publication_complete_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f'))
    dossier.mkdir(parents=True)
    noms = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--', 'TP3'],
                                  cwd=ROOT.parent, text=True).splitlines()
    for nom in noms:
        source = ROOT.parent / nom
        if source.is_file():
            cible = dossier / source.relative_to(ROOT)
            cible.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, cible)
    shutil.copy2(ROOT / 'verification_execution.json', dossier / 'verification_execution.json')
    shutil.copy2(ROOT.parent / 'README.md', dossier / 'README_depot.md')
    archive = ROOT / '.cache' / 'TP3_verification_ChatGPT.zip'
    if archive.exists():
        cible = ROOT / '.cache' / 'TP3_verification_ChatGPT_complete.zip'
        if not cible.exists():
            shutil.copy2(archive, cible)
    return dossier


def preparer():
    sauvegarde = sauvegarder()
    rapport = lire(ROOT / 'resultats_tp3.json')
    modifies = []
    for p in [ROOT / 'resultats_tp3.json', *(ROOT / 'evaluations').rglob('*.json')]:
        if p.name in {'empreintes.json', 'verification_publication.json'}:
            continue
        original = lire(p)
        public = masquer(original)
        verifier_structure(original, public)
        if 'cas' in public and isinstance(public['cas'], list) and public['cas'] and 'agent' in public['cas'][0]:
            public['publication'] = {'note': NOTE, 'textes_et_figures_reproduits': False}
        if public != original:
            ecrire(p, public)
        if public != original or 'publication' in public:
            modifies.append(p.relative_to(ROOT).as_posix())

    p = ROOT / 'TP3_RAG_Multimodal_Agentique.ipynb'
    notebook = nbformat.read(p, as_version=4)
    original = copy.deepcopy(notebook)
    sorties_retirees = copy.deepcopy(notebook.metadata.get('publication', {}).get('sorties_retirees', []))
    modes = {'Texte → texte', 'Texte → image', 'Image → texte', 'Image → image', 'Recherche mixte', 'Hybride texte + image'}
    for i, cellule in enumerate(notebook.cells):
        cellule.pop('attachments', None)
        if cellule.cell_type != 'code':
            continue
        if i == 3:
            cellule.outputs = [nbformat.v4.new_output('stream', name='stdout', text=
                f"Pages PDF : {rapport['corpus']['pages']}\nImages intégrées uniques : {rapport['corpus']['images_integrees_uniques']}\nFigures indexées : {rapport['corpus']['images']}\n{MASQUE}\n")]
            sorties_retirees.append({'cellule': i, 'elements': 'aperçu textuel et figures du PDF'})
            continue
        if i == 9:
            lignes = []
            for sortie in cellule.outputs:
                for ligne in sortie.get('text', '').splitlines():
                    if ligne in modes or (' | cosinus ' in ligne and ' | page ' in ligne) or ligne.startswith('Scores par entrée :'):
                        lignes.append(ligne)
            cellule.outputs = [nbformat.v4.new_output('stream', name='stdout', text='\n'.join(lignes) + '\n' + MASQUE + '\n')]
            sorties_retirees.append({'cellule': i, 'elements': 'passages récupérés ; rangs et scores conservés'})
            continue
        gardees = []
        for sortie in cellule.outputs:
            data = sortie.get('data', {})
            reproduite = any(k.startswith('image/') and k != 'image/svg+xml' for k in data)
            if 'image/svg+xml' in data:
                svg = data['image/svg+xml']
                assert svg in [(ROOT / 'docs' / n).read_text(encoding='utf-8') for n in ['agent.svg', 'architecture.svg']]
            if reproduite or (i in {11, 13, 19} and 'text/markdown' in data):
                gardees.append(nbformat.v4.new_output('stream', name='stdout', text=MASQUE + '\n'))
                sorties_retirees.append({'cellule': i, 'elements': 'réponse documentaire ou image reproduite'})
            else:
                if i == 13 and sortie.output_type == 'stream':
                    sortie.text = '\n'.join('consulter_pdf [requête textuelle masquée]' + ligne[ligne.index(' | sources :'):] if ligne.startswith('consulter_pdf ') and ' | sources :' in ligne else ligne for ligne in sortie.text.splitlines()) + '\n'
                gardees.append(sortie)
        cellule.outputs = gardees
    for a, b in zip(original.cells, notebook.cells):
        if a.cell_type == 'code':
            assert a.source == b.source and a.execution_count == b.execution_count
    avertissement = '\n\n**Édition publique :** ' + NOTE + ' Les sorties complètes se régénèrent localement en exécutant le code. Voir [PUBLICATION.md](PUBLICATION.md).'
    if '**Édition publique :**' not in notebook.cells[0].source:
        notebook.cells[0].source += avertissement
    sorties_retirees = list({(x['cellule'], x['elements']): x for x in sorties_retirees}.values())
    notebook.metadata['publication'] = {'edition': 'publique', 'experiences_reexecutees': False,
        'notebook_complet_sha256': sha(sauvegarde / p.name), 'sorties_retirees': sorties_retirees}
    nbformat.validate(notebook)
    p.write_text(nbformat.writes(notebook), encoding='utf-8', newline='\n')
    instructions = ROOT / 'evaluations/reference_initiale/instructions_agent.txt'
    instructions.write_text(instructions.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    ecrire(ROOT / 'evaluations' / 'reference_initiale' / 'empreintes.json', {
        'portee': 'Empreintes des fichiers publics ; les originaux complets sont conservés localement.',
        'fichiers': {f.name: sha(f) for f in sorted((ROOT / 'evaluations' / 'reference_initiale').iterdir())
                     if f.is_file() and f.name not in {'empreintes.json', 'verification_execution.json'}}})
    relecture = ROOT / 'evaluations' / 'relecture_semantique.json'
    analyse = lire(relecture)
    analyse['rapport_complet_local_sha256'] = sha(sauvegarde / 'resultats_tp3.json')
    analyse['rapport_final_sha256'] = sha(ROOT / 'resultats_tp3.json')
    ecrire(relecture, analyse)
    ecrire(ROOT / 'publication.json', {'edition': 'publique', 'note': NOTE,
        'notebook_sha256': sha(p), 'notebook_complet_local_sha256': sha(sauvegarde / p.name),
        'rapport_sha256': sha(ROOT / 'resultats_tp3.json'),
        'rapport_complet_local_sha256': sha(sauvegarde / 'resultats_tp3.json'),
        'sorties_retirees': sorties_retirees, 'rapports_traites': modifies,
        'metriques_et_provenance_preservees': True})


def verifier():
    publication = lire(ROOT / 'publication.json')
    assert sha(ROOT / 'TP3_RAG_Multimodal_Agentique.ipynb') == publication['notebook_sha256']
    assert sha(ROOT / 'resultats_tp3.json') == publication['rapport_sha256']
    notebook = nbformat.read(ROOT / 'TP3_RAG_Multimodal_Agentique.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cellule in notebook.cells:
        assert not cellule.get('attachments')
        for sortie in cellule.get('outputs', []):
            assert not any(k.startswith('image/') and k != 'image/svg+xml' for k in sortie.get('data', {}))
    for p in [ROOT / 'resultats_tp3.json', *(ROOT / 'evaluations').glob('essai_*.json'), ROOT / 'evaluations/reference_initiale/resultats_tp3.json']:
        assert lire(p) == masquer(lire(p)), p
    print('Version publique vérifiée : notebook valide, reproductions retirées, métriques et provenance conservées.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Sauvegarder la copie complète et retirer les reproductions.')
    args = parser.parse_args()
    if args.apply:
        preparer()
    verifier()
