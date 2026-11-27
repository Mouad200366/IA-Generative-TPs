"""Exécution depuis un noyau neuf, avec sauvegarde de chaque cellule terminée."""
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

ROOT=Path(__file__).resolve().parent


def main():
    # Le tokenizer GPT-2 affiche notamment Ġ, absent de certaines consoles Windows.
    for flux in (sys.stdout,sys.stderr):
        if hasattr(flux,'reconfigure'):
            flux.reconfigure(encoding='utf-8',errors='backslashreplace')
    path=ROOT/'TP4_Architecture_Transformer.ipynb'
    nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':
            ast.parse(c.source);c.outputs=[];c.execution_count=None
    preuve={'statut':'en_cours','noyau_neuf':True,'debut_utc':datetime.now(timezone.utc).isoformat(),'cellules_executees':[]}
    def sauver():
        # Anonymiser les chemins d'avertissements sans effacer leur message.
        for c in nb.cells:
            for o in c.get('outputs',[]):
                if o.output_type=='stream':o.text=o.text.replace(str(ROOT),'<TP4>').replace(ROOT.as_posix(),'<TP4>')
        path.write_text(nbformat.writes(nb),encoding='utf-8',newline='\n')
        (ROOT/'verification_execution.json').write_text(json.dumps(preuve,indent=2,ensure_ascii=False),encoding='utf-8')
    def terminee(cell,cell_index,**kwargs):
        preuve['cellules_executees'].append({'index':cell_index,'compteur':cell.execution_count})
        sauver();print('Cellule',cell_index,'terminée, compteur',cell.execution_count,flush=True)
        for o in cell.get('outputs',[]):
            if o.output_type=='stream':print(o.text,end='',flush=True)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with tempfile.TemporaryDirectory(prefix='tp4-kernel-') as temp:
        kernels=Path(temp)/'kernels'/'tp4-uv';kernels.mkdir(parents=True)
        (kernels/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
            'display_name':'TP4 UV temporaire','language':'python'}),encoding='utf-8')
        os.environ['JUPYTER_PATH']=temp+os.pathsep+os.environ.get('JUPYTER_PATH','')
        try:
            NotebookClient(nb,kernel_name='tp4-uv',timeout=1800,resources={'metadata':{'path':str(ROOT)}},on_cell_executed=terminee).execute()
        except Exception as exc:
            preuve.update(statut='echec',erreur=type(exc).__name__);sauver();raise
    codes=[c for c in nb.cells if c.cell_type=='code']
    assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    r=json.loads((ROOT/'resultats_tp4.json').read_text(encoding='utf-8'))
    preuve.update(statut='succes',fin_utc=datetime.now(timezone.utc).isoformat(),cellules_python=len(codes),
        erreurs_notebook=0,validation=r['validation'])
    sauver();preuve['notebook_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (ROOT/'verification_execution.json').write_text(json.dumps(preuve,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(preuve,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
