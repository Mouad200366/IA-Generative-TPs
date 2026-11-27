"""Boucle commune avec reprise par époque et choix des poids sur validation seulement."""
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from donnees import corrompre_mlm

ROOT = Path(__file__).resolve().parent


def ecrire_json(path, objet):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporaire=path.with_suffix(path.suffix+'.tmp')
    temporaire.write_text(json.dumps(objet,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')
    temporaire.replace(path)


def empreinte(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def initialiser(seed=42):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def sauvegarder_torch(path,objet):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(objet,tmp);tmp.replace(path)


def avancement(nom,ep,total,statut='en_cours'):
    p=ROOT/'.cache/avancement.json'
    r=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    r[nom]={'epoque':ep,'total':total,'statut':statut,'date_utc':datetime.now(timezone.utc).isoformat()}
    ecrire_json(p,r)
    if statut=='termine' or ep==1 or ep%5==0:
        with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
            f.write(f'\n- {r[nom]["date_utc"]} : {nom}, époque {ep}/{total}, {statut}.')


def entrainer(nom,modele,train,val,config,device):
    configuration={**config,'device':str(device),'architecture_sha256':empreinte(ROOT/'modeles.py')}
    signature=hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()
    poids=ROOT/'poids'/f'{nom}.pt'
    resultat_path=ROOT/'.cache'/f'{nom}_entrainement.json'
    checkpoint=ROOT/'.cache/checkpoints'/f'{nom}.pt'
    if resultat_path.exists() and poids.exists():
        resultat=json.loads(resultat_path.read_text(encoding='utf-8'))
        if resultat['signature']!=signature:
            raise ValueError(f'Configuration modifiée pour {nom} : ne pas réentraîner implicitement.')
        assert empreinte(poids)==resultat['poids_sha256']
        modele.load_state_dict(torch.load(poids,weights_only=True,map_location='cpu'))
        modele.to(device).eval()
        print(nom,': entraînement terminé réutilisé.',flush=True)
        return resultat
    modele.to(device)
    optimiseur=torch.optim.Adam(modele.parameters(),lr=config['lr'])
    critere=nn.MSELoss() if config['tache']=='regression' else nn.CrossEntropyLoss(ignore_index=-100)
    corruption=torch.Generator().manual_seed(4242)
    histoire={'train':[],'validation':[],'accuracy_validation':[],'secondes_train':[],'secondes_validation':[]}
    cumul={'selectionnes':0,'ordinaires':0,'branche_mask':0,'branche_aleatoire':0,'branche_inchangee':0}
    debut_epoque=0;meilleure=float('inf');epoque_choisie=0
    if checkpoint.exists():
        etat=torch.load(checkpoint,weights_only=False,map_location='cpu')
        if etat['signature']!=signature:raise ValueError('Checkpoint incompatible ; aucune reprise silencieuse.')
        modele.load_state_dict(etat['modele']);optimiseur.load_state_dict(etat['optimiseur'])
        histoire=etat['histoire'];cumul=etat['cumul'];debut_epoque=etat['epoque']
        meilleure=etat['meilleure'];epoque_choisie=etat['epoque_choisie']
        torch.set_rng_state(etat['rng_torch']);np.random.set_state(etat['rng_numpy']);random.setstate(etat['rng_python'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(etat['rng_cuda'])
        train.generator.set_state(etat['rng_loader']);corruption.set_state(etat['rng_corruption'])
        print(nom,': reprise après époque',debut_epoque,flush=True)

    def lot_calcul(lot,apprendre):
        if config['tache']=='bert':
            if apprendre:
                originals=lot[0]
                x,y,_,stats=corrompre_mlm(originals,config['vocabulaire'],corruption)
                for k,v in stats.items():cumul[k]+=v
                padding=originals.eq(0)
            else:x,y,padding=lot
            x=x.to(device);y=y.to(device);padding=padding.to(device)
            logits=modele(x,padding);actifs=y.ne(-100)
            perte=critere(logits.flatten(0,1),y.flatten())
            return perte,int(actifs.sum()),int((logits.argmax(-1)[actifs]==y[actifs]).sum())
        x,y=[v.to(device) for v in lot]
        logits=modele(x)
        if config['tache']=='gpt':
            perte=critere(logits.flatten(0,1),y.flatten())
            return perte,y.numel(),int((logits.argmax(-1)==y).sum())
        return critere(logits,y),y.numel(),0

    def synchroniser():
        if device.type=='cuda':torch.cuda.synchronize()

    for ep in range(debut_epoque,config['epoques']):
        modele.train();synchroniser();debut=time.perf_counter();somme=0.;nombre=0
        for lot in train:
            optimiseur.zero_grad(set_to_none=True)
            perte,n,_=lot_calcul(lot,True)
            if not torch.isfinite(perte):raise FloatingPointError('Perte non finie.')
            perte.backward();optimiseur.step()
            somme+=float(perte.detach())*n;nombre+=n
        synchroniser();histoire['secondes_train'].append(time.perf_counter()-debut)
        histoire['train'].append(somme/nombre)
        score=histoire['train'][-1]
        if val is not None:
            modele.eval();synchroniser();debut=time.perf_counter();somme=0.;nombre=0;corrects=0
            with torch.inference_mode():
                for lot in val:
                    perte,n,c=lot_calcul(lot,False)
                    somme+=float(perte)*n;nombre+=n;corrects+=c
            synchroniser();histoire['secondes_validation'].append(time.perf_counter()-debut)
            score=somme/nombre;histoire['validation'].append(score)
            histoire['accuracy_validation'].append(corrects/nombre if config['tache']!='regression' else None)
        retenir=(val is not None and score<meilleure) or (val is None and ep+1==config['epoques'])
        if retenir:
            meilleure=score;epoque_choisie=ep+1
            sauvegarder_torch(poids,{k:v.detach().cpu() for k,v in modele.state_dict().items()})
        if (ep+1)%config.get('checkpoint_toutes',1)==0 or ep+1==config['epoques']:
            sauvegarder_torch(checkpoint,{'signature':signature,'modele':modele.state_dict(),'optimiseur':optimiseur.state_dict(),
                'epoque':ep+1,'histoire':histoire,'cumul':cumul,'meilleure':meilleure,'epoque_choisie':epoque_choisie,
                'rng_torch':torch.get_rng_state(),'rng_numpy':np.random.get_state(),'rng_python':random.getstate(),
                'rng_cuda':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                'rng_loader':train.generator.get_state(),'rng_corruption':corruption.get_state()})
            avancement(nom,ep+1,config['epoques'])
        if ep==0 or (ep+1)%max(1,config['epoques']//5)==0:
            print(f'{nom} {ep+1}/{config["epoques"]} : train={histoire["train"][-1]:.6f}, suivi={score:.6f}',flush=True)
    modele.load_state_dict(torch.load(poids,weights_only=True,map_location='cpu'));modele.eval()
    resultat={'signature':signature,'configuration':configuration,'histoire':histoire,'epoque_choisie':epoque_choisie,
        'selection':'minimum de perte validation' if val is not None else 'dernière époque imposée',
        'meilleure_perte_suivie':meilleure,'parametres':sum(p.numel() for p in modele.parameters()),
        'secondes_entrainement':sum(histoire['secondes_train']),'secondes_validation':sum(histoire['secondes_validation']),
        'poids':poids.relative_to(ROOT).as_posix(),'poids_sha256':empreinte(poids),'corruption_train':cumul,
        'date_fin_utc':datetime.now(timezone.utc).isoformat()}
    ecrire_json(resultat_path,resultat);avancement(nom,config['epoques'],config['epoques'],'termine')
    return resultat
