"""Exécuter le TP2 depuis un noyau neuf et écrire une preuve locale de vérification."""
import ast
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    print('Préparation du notebook et contrôle de syntaxe...', flush=True)
    dossier = Path(__file__).resolve().parent
    notebook_path = dossier / 'TP2_Modeles_Generatifs.ipynb'
    preuve_path = dossier / 'verification_execution.json'
    nb = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            cell.outputs = []
            cell.execution_count = None

    preuve = {'debut_utc': datetime.now(timezone.utc).isoformat(),
              'noyau_neuf': True, 'statut': 'en_cours', 'cellules_executees': []}

    def sauvegarder():
        nb.metadata.kernelspec = {'display_name': 'Python 3.11 (TP2 UV)',
                                  'language': 'python', 'name': 'python3'}
        nbformat.write(nb, notebook_path)
        preuve_path.write_text(json.dumps(preuve, indent=2, ensure_ascii=False), encoding='utf-8')

    def cellule_executee(cell, cell_index, **kwargs):
        preuve['cellules_executees'].append({'index': cell_index, 'compteur': cell.execution_count})
        sauvegarder()
        print(f'Cellule {cell_index} exécutée (compteur {cell.execution_count})', flush=True)
        for output in cell.get('outputs', []):
            if output.output_type == 'stream':
                print(output.text, end='', flush=True)

    # Le kernelspec reste temporaire et n'est jamais publié avec un chemin personnel.
    with tempfile.TemporaryDirectory(prefix='tp2-kernel-') as temp:
        kernels = Path(temp) / 'kernels' / 'tp2-uv'
        kernels.mkdir(parents=True)
        (kernels / 'kernel.json').write_text(json.dumps({
            'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'TP2 UV temporaire', 'language': 'python',
        }), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = temp + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        os.environ.setdefault('MPLBACKEND', 'module://matplotlib_inline.backend_inline')
        client = NotebookClient(nb, timeout=1800, kernel_name='tp2-uv',
                                resources={'metadata': {'path': str(dossier)}},
                                on_cell_executed=cellule_executee)
        try:
            print('Démarrage du noyau neuf TP2 UV...', flush=True)
            client.execute()
        except Exception as exc:
            preuve['statut'] = 'echec'
            preuve['type_erreur'] = type(exc).__name__
            sauvegarder()
            raise

    codes = [c for c in nb.cells if c.cell_type == 'code']
    assert all(not any(o.output_type == 'error' for o in c.outputs) for c in codes)
    assert [c.execution_count for c in codes] == list(range(1, len(codes) + 1))
    resultats = json.loads((dossier / 'resultats_tp2.json').read_text(encoding='utf-8'))
    assert all(x['identique'] and x['images_test'] == 10000 for x in resultats['rechargement'].values())
    preuve.update({'statut': 'succes', 'fin_utc': datetime.now(timezone.utc).isoformat(),
                   'nombre_cellules_python': len(codes),
                   'graphiques_integres': sum('image/png' in o.get('data', {}) for c in codes for o in c.outputs),
                   'erreurs': 0, 'compteurs_consecutifs': True,
                   'rechargement': resultats['rechargement'],
                   'environnement': resultats['environnement'],
                   'sha256_poids': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((dossier / 'poids').glob('*.pth'))}})
    sauvegarder()
    preuve['sha256_notebook_execute'] = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    preuve_path.write_text(json.dumps(preuve, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(preuve, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
