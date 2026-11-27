"""Contrats demandés par le support : causalité, padding, cibles et absence de fuite."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from donnees import caracteres, corrompre_mlm, fenetres_horaires, heures_valides, phrases_bert, sinus
from entrainement import initialiser
from modeles import SimpleRNN, SimpleLSTM, MiniTransformer, EnergyTransformer, MiniGPT, MiniBERT


@pytest.fixture(autouse=True)
def graines():
    initialiser()


@pytest.mark.parametrize('classe,forme,sortie,parametres',[
    (SimpleRNN,(3,10,1),(3,1),321),(SimpleLSTM,(3,10,1),(3,1),1233),
    (MiniTransformer,(3,10,1),(3,1),2433),(EnergyTransformer,(3,24,1),(3,1),26273)])
def test_architectures_series(classe,forme,sortie,parametres):
    modele=classe().eval()
    assert sum(p.numel() for p in modele.parameters())==parametres
    with torch.inference_mode():assert tuple(modele(torch.zeros(forme)).shape)==sortie


def test_sinus_cible_suivante_et_decoupage():
    _,serie,x,y=sinus()
    assert x.shape==(190,10,1) and y.shape==(190,1)
    assert torch.equal(x[0,:,0],serie[:10])
    assert torch.equal(y[:,0],serie[10:])
    assert len(x[:150])==150 and len(x[150:])==40


def test_fenetres_energie_consecutives_sans_fuite():
    index=pd.date_range('2020-01-01',periods=1000,freq='h')
    serie=pd.Series(np.arange(1000,dtype=float),index=index)
    serie.iloc[50]=np.nan;serie.iloc[950]=np.nan
    x,y,indices,masks,meta=fenetres_horaires(serie)
    assert not any(50<=i<=74 or 950<=i<=974 for i in indices)
    assert np.all(indices[masks['train']]<700)
    assert np.all((indices[masks['validation']]>=700)&(indices[masks['validation']]<850))
    assert np.all(indices[masks['test']]>=850)
    autre=serie.copy();autre.iloc[700:]=999999
    autres=fenetres_horaires(autre)[-1]
    assert autres['moyenne_train_kw']==meta['moyenne_train_kw']
    assert autres['ecart_type_train_kw']==meta['ecart_type_train_kw']
    assert x.shape[1:]==(24,1) and y.shape[1:]==(1,)


def test_gpt_decalage_et_frontiere():
    x,y,xv,yv,stoi,meta=caracteres()
    texte=(Path(__file__).resolve().parents[1]/'corpus.txt').read_text(encoding='utf-8')
    assert torch.equal(x[:,1:],y[:,:-1]) and torch.equal(xv[:,1:],yv[:,:-1])
    assert int(xv[0,0])==stoi[texte[meta['frontiere_train']]]
    assert set(stoi)==set(texte[:meta['frontiere_train']])


def test_seuil_horaire_45_mesures():
    puissance=pd.Series(np.ones(120),index=pd.date_range('2020-01-01',periods=120,freq='min'))
    puissance.iloc[:16]=np.nan;puissance.iloc[60:75]=np.nan
    horaire=puissance.resample('h').agg(['mean','count'])
    resultat=heures_valides(horaire)
    assert horaire['count'].tolist()==[44,45]
    assert pd.isna(resultat.iloc[0]) and resultat.iloc[1]==1


def test_gpt_causal_et_formes():
    modele=MiniGPT(30).eval();x=torch.randint(0,30,(2,64));autre=x.clone();autre[:,40]=(autre[:,40]+1)%30
    with torch.inference_mode():a=modele(x);b=modele(autre);r=modele.representations(x)
    assert a.shape==(2,64,30) and r.shape==(2,64,64)
    assert torch.allclose(a[:,:40],b[:,:40],atol=1e-6,rtol=0)


def test_vocabulaire_bert_train_uniquement():
    train,val,voc,phrases,meta=phrases_bert()
    assert [voc[k] for k in ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]']]==list(range(5))
    assert meta['phrases_communes']==0 and train.shape[1]==val.shape[1]==32
    assert set(meta['indices_train']).isdisjoint(meta['indices_validation'])
    assert meta['tokens_unk_validation']==int(val.eq(1).sum())


def test_mlm_labels_80_10_10_et_specials():
    ids=torch.randint(5,50,(1000,32));ids[:,0]=2;ids[:,-1]=3
    ids[0]=0;ids[0,0]=2;ids[0,1]=3
    a,labels,sel,stats=corrompre_mlm(ids,50,torch.Generator().manual_seed(42))
    b,labels2,sel2,_=corrompre_mlm(ids,50,torch.Generator().manual_seed(42))
    assert torch.equal(a,b) and torch.equal(labels,labels2) and torch.equal(sel,sel2)
    assert not sel[ids<5].any() and sel[1:].any(dim=1).all() and not sel[0].any()
    assert torch.equal(labels[sel],ids[sel]) and labels[~sel].eq(-100).all()
    assert sel[(a==ids)&sel].any()
    n=stats['selectionnes']
    assert .13<n/stats['ordinaires']<.18
    for k,p in [('branche_mask',.8),('branche_aleatoire',.1),('branche_inchangee',.1)]:
        assert abs(stats[k]/n-p)<.025


def test_bert_padding_et_contexte_droit():
    modele=MiniBERT(50).eval();x=torch.randint(5,50,(2,32));x[:,17:]=0
    padding=x.eq(0);autre=x.clone();autre[padding]=49
    futur=x.clone();futur[:,10]=6
    with torch.inference_mode():a=modele(x,padding);b=modele(autre,padding);c=modele(futur,padding)
    assert a.shape==(2,32,50)
    assert torch.allclose(a[~padding],b[~padding],atol=1e-6,rtol=0)
    assert float((a[:,2]-c[:,2]).abs().max())>1e-7


def test_rapport_et_poids_entraines():
    root=Path(__file__).resolve().parents[1];p=root/'resultats_tp4.json'
    if not p.exists():pytest.skip('Le notebook doit être exécuté pour ce contrôle final.')
    r=json.loads(p.read_text(encoding='utf-8'))
    assert len(r['sinus']['modeles'])==3
    for item in [*r['sinus']['modeles'].values(),r['energie'],r['mini_gpt'],r['mini_bert']]:
        assert (root/item['poids']).is_file()
    assert r['mini_gpt']['tests']['causalite']['reussi']
    assert r['mini_bert']['tests']['padding']['reussi']
    assert all(g['nouveaux_caracteres']==200 and len(g['texte_nouveau'])==200 for g in r['mini_gpt']['generations'])
