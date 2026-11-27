"""Exécuter le rendu depuis un noyau Python neuf et produire la preuve locale."""
import ast
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone

import nbformat
from nbclient import NotebookClient

from tp3rag.config import ROOT, ecrire_json, lire_json


def main():
    path = ROOT / 'TP3_RAG_Multimodal_Agentique.ipynb'
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == 'code':
            ast.parse(c.source)
            c.outputs = []
            c.execution_count = None
    preuve = {'debut_utc': datetime.now(timezone.utc).isoformat(), 'noyau_neuf': True,
              'statut': 'en_cours', 'cellules_executees': []}

    def sauver():
        nb.metadata.kernelspec = {'display_name': 'Python 3.11 (TP3 UV)', 'language': 'python', 'name': 'python3'}
        nbformat.write(nb, path)
        ecrire_json(ROOT / 'verification_execution.json', preuve)

    def executee(cell, cell_index, **kwargs):
        preuve['cellules_executees'].append({'index': cell_index, 'compteur': cell.execution_count})
        sauver()
        print('Cellule', cell_index, 'exécutée, compteur', cell.execution_count, flush=True)
        for o in cell.get('outputs', []):
            if o.output_type == 'stream':
                print(o.text, end='', flush=True)

    with tempfile.TemporaryDirectory(prefix='tp3-kernel-') as temp:
        from pathlib import Path
        dossier = Path(temp) / 'kernels' / 'tp3-uv'
        dossier.mkdir(parents=True)
        (dossier / 'kernel.json').write_text(json.dumps({'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'TP3 UV temporaire', 'language': 'python'}), encoding='utf-8')
        os.environ['JUPYTER_PATH'] = temp + os.pathsep + os.environ.get('JUPYTER_PATH', '')
        client = NotebookClient(nb, timeout=1800, kernel_name='tp3-uv',
            resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=executee)
        try:
            client.execute()
        except Exception as exc:
            preuve.update({'statut': 'echec', 'type_erreur': type(exc).__name__})
            sauver()
            raise
    codes = [c for c in nb.cells if c.cell_type == 'code']
    assert [c.execution_count for c in codes] == list(range(1, len(codes) + 1))
    assert not any(o.output_type == 'error' for c in codes for o in c.outputs)
    rapport = lire_json(ROOT / 'resultats_tp3.json')
    preuve.update({'statut': 'succes', 'fin_utc': datetime.now(timezone.utc).isoformat(),
        'cellules_python': len(codes), 'erreurs_notebook': 0,
        'validation_pipeline': rapport['statut'], 'mesures': rapport['mesures'],
        'mcp': rapport['mcp']['statut'], 'streamlit': rapport['streamlit']['statut']})
    sauver()
    preuve['sha256_notebook'] = hashlib.sha256(path.read_bytes()).hexdigest()
    ecrire_json(ROOT / 'verification_execution.json', preuve)
    print(json.dumps(preuve, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
