"""Préparations sans fuite temporelle et corruption MLM contrôlable."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from modeles import PAD, UNK, CLS, SEP, MASK

ROOT = Path(__file__).resolve().parent


def chargeur(x, y, batch=64, melanger=False):
    return DataLoader(TensorDataset(x,y),batch_size=batch,shuffle=melanger,
        generator=torch.Generator().manual_seed(42),num_workers=0)


def sinus():
    periode=torch.linspace(0,20,200)
    serie=torch.sin(periode)
    x=torch.stack([serie[i:i+10] for i in range(190)]).unsqueeze(-1)
    y=serie[10:].unsqueeze(-1)
    return periode,serie,x,y


def fenetres_horaires(serie, fenetre=24):
    """Ne jamais supprimer les NaN avant le fenêtrage : les heures restent consécutives."""
    n=len(serie); fin_train=int(.70*n); fin_val=int(.85*n)
    moyenne=float(serie.iloc[:fin_train].mean())
    ecart=float(serie.iloc[:fin_train].std())
    if not np.isfinite(ecart) or ecart<=0:
        raise ValueError('Écart-type train non valide.')
    assert np.all((serie.index[1:]-serie.index[:-1])==pd.Timedelta(hours=1))
    valeurs=((serie-moyenne)/ecart).to_numpy()
    toutes=np.lib.stride_tricks.sliding_window_view(valeurs,fenetre+1)
    valides=np.isfinite(toutes).all(axis=1)
    x=torch.tensor(toutes[valides,:fenetre].copy(),dtype=torch.float32).unsqueeze(-1)
    y=torch.tensor(toutes[valides,fenetre].copy(),dtype=torch.float32).unsqueeze(-1)
    indices=np.arange(fenetre,n)[valides]
    masks={'train':indices<fin_train,'validation':(indices>=fin_train)&(indices<fin_val),'test':indices>=fin_val}
    return x,y,indices,masks,{'moyenne_train_kw':moyenne,'ecart_type_train_kw':ecart,'ddof':1,
        'heures_total':n,'heures_manquantes':int(serie.isna().sum()),'fin_train':fin_train,'fin_validation':fin_val,
        'fenetres':{k:int(v.sum()) for k,v in masks.items()},'frontieres_cibles':[str(serie.index[fin_train]),str(serie.index[fin_val])]}


def heures_valides(horaire):
    return horaire['mean'].where(horaire['count']>=45)


def energie():
    fichier=ROOT/'data/household_power_consumption.txt'
    with fichier.open('rb') as source:
        digest=hashlib.file_digest(source,'sha256').hexdigest()
    cache=ROOT/'.cache/energie_horaire.csv'; info=ROOT/'.cache/energie_source.json'
    if cache.exists() and info.exists() and json.loads(info.read_text())['sha256']==digest:
        horaire=pd.read_csv(cache,index_col=0,parse_dates=True)
    else:
        df=pd.read_csv(fichier,sep=';',na_values=['?',''],usecols=['Date','Time','Global_active_power'],low_memory=False)
        dates=pd.to_datetime(df['Date']+' '+df['Time'],format='%d/%m/%Y %H:%M:%S')
        puissance=pd.to_numeric(df['Global_active_power'],errors='coerce')
        puissance.index=pd.DatetimeIndex(dates)
        horaire=puissance.resample('h').agg(['mean','count'])
        cache.parent.mkdir(exist_ok=True)
        horaire.to_csv(cache)
        info.write_text(json.dumps({'sha256':digest,'lignes_minutes':len(df),'valeurs_minutes_manquantes':int(puissance.isna().sum())}),encoding='utf-8')
    serie=heures_valides(horaire)
    x,y,indices,masks,meta=fenetres_horaires(serie)
    meta.update(json.loads(info.read_text()))
    meta['minimum_mesures_valides_par_heure']=45
    return serie,x,y,indices,masks,meta


def caracteres():
    texte=(ROOT/'corpus.txt').read_text(encoding='utf-8')
    frontiere=int(.9*len(texte))
    vocab=sorted(set(texte[:frontiere]))
    inconnus=set(texte[frontiere:])-set(vocab)
    if inconnus:
        raise ValueError(f'Caractères de validation hors alphabet train : {inconnus}')
    stoi={c:i for i,c in enumerate(vocab)}
    tokens=torch.tensor([stoi[c] for c in texte],dtype=torch.long)
    def paires(segment,pas):
        debuts=range(0,len(segment)-64,pas)
        return torch.stack([segment[i:i+64] for i in debuts]),torch.stack([segment[i+1:i+65] for i in range(0,len(segment)-64,pas)])
    x,y=paires(tokens[:frontiere],16); xv,yv=paires(tokens[frontiere:],64)
    return x,y,xv,yv,stoi,{'caracteres_total':len(texte),'taille_vocabulaire':len(vocab),'frontiere_train':frontiere,
        'fenetres_train':len(x),'fenetres_validation':len(xv),'longueur':64,'pas_train':16,'pas_validation':64,
        'vocabulaire_construit_sur_train':True,'fenetres_traversant_frontiere':0}


def mots(texte):
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)*",texte.lower(),re.UNICODE)


def phrases_bert():
    phrases=[s.strip() for s in (ROOT/'corpus_bert.txt').read_text(encoding='utf-8').splitlines() if s.strip()]
    ordre=np.random.default_rng(42).permutation(len(phrases))
    frontiere=int(.8*len(phrases)); train_idx=ordre[:frontiere]; val_idx=ordre[frontiere:]
    voc={'[PAD]':PAD,'[UNK]':UNK,'[CLS]':CLS,'[SEP]':SEP,'[MASK]':MASK}
    voc.update({m:i+5 for i,m in enumerate(sorted({m for j in train_idx for m in mots(phrases[j])}))})
    def encoder(indices):
        sequences=[]
        for j in indices:
            ids=[CLS]+[voc.get(m,UNK) for m in mots(phrases[j])[:30]]+[SEP]
            sequences.append(ids+[PAD]*(32-len(ids)))
        return torch.tensor(sequences,dtype=torch.long)
    train=encoder(train_idx); val=encoder(val_idx)
    ordinaires=(val==UNK)|(val>=5)
    meta={'phrases_total':len(phrases),'phrases_train':len(train),'phrases_validation':len(val),
        'taille_vocabulaire':len(voc),'tokens_ordinaires_validation':int(ordinaires.sum()),
        'tokens_unk_validation':int((val==UNK).sum()),'taux_unk_validation':float((val==UNK).sum()/ordinaires.sum()),
        'indices_train':train_idx.tolist(),'indices_validation':val_idx.tolist(),
        'vocabulaire_train_seulement':True,'phrases_communes':len(set(phrases[j] for j in train_idx)&set(phrases[j] for j in val_idx))}
    return train,val,voc,phrases,meta


def corrompre_mlm(ids, taille_vocabulaire, generator):
    """Labels tirés de l'original, y compris pour les 10 % sélectionnés inchangés."""
    ordinaires=ids>=5
    selection=(torch.rand(ids.shape,generator=generator)<.15)&ordinaires
    for i in range(len(ids)):
        positions=ordinaires[i].nonzero().flatten()
        if len(positions) and not selection[i].any():
            selection[i,positions[torch.randint(len(positions),(1,),generator=generator)]]=True
    choix=torch.rand(ids.shape,generator=generator)
    masques=selection&(choix<.8)
    aleatoires=selection&(choix>=.8)&(choix<.9)
    inchanges=selection&(choix>=.9)
    corrompus=ids.clone(); labels=ids.clone(); labels[~selection]=-100
    corrompus[masques]=MASK
    tirages=torch.randint(5,taille_vocabulaire,ids.shape,generator=generator)
    corrompus[aleatoires]=tirages[aleatoires]
    return corrompus,labels,selection,{'selectionnes':int(selection.sum()),'ordinaires':int(ordinaires.sum()),
        'branche_mask':int(masques.sum()),'branche_aleatoire':int(aleatoires.sum()),'branche_inchangee':int(inchanges.sum())}
