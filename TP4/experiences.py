"""Expériences du TP4 : mêmes données et métriques, sans entraînements redondants."""
import gc
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import donnees
from entrainement import ROOT, ecrire_json, empreinte, entrainer, initialiser
from modeles import SimpleRNN, SimpleLSTM, MiniTransformer, EnergyTransformer, MiniGPT, MiniBERT, PAD, MASK


def metriques(y,p):
    erreur=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    mse=float(np.mean(erreur**2))
    return {'mse':mse,'mae':float(np.mean(np.abs(erreur))),'rmse':math.sqrt(mse),'cibles':len(erreur)}


def figure(nom):
    (ROOT/'figures').mkdir(exist_ok=True)
    plt.tight_layout();plt.savefig(ROOT/'figures'/nom,dpi=150,bbox_inches='tight');plt.close()


def courbes(histoire,nom,titre):
    plt.figure(figsize=(8,4));plt.plot(histoire['train'],label='Entraînement')
    if histoire['validation']:plt.plot(histoire['validation'],label='Validation fixe' if 'bert' in nom else 'Validation')
    plt.xlabel('Époque (indice à partir de 0)');plt.ylabel('Perte');plt.title(titre);plt.legend();plt.grid(alpha=.2)
    figure(nom)


def rechargement(modele,constructeur,arguments,poids,device):
    copie=constructeur().to(device).eval()
    copie.load_state_dict(torch.load(ROOT/poids,weights_only=True,map_location=device))
    modele.eval()
    with torch.inference_mode():a=modele(*arguments);b=copie(*arguments)
    delta=float((a-b).abs().max())
    assert delta<=1e-6
    return {'reussi':True,'ecart_max':delta,'tolerance':1e-6,'identique':bool(torch.equal(a,b))}


def sinus():
    periode,serie,x,y=donnees.sinus()
    plt.figure(figsize=(9,3));plt.plot(periode,serie,label='sin(x)')
    plt.axvline(float(periode[160]),color='#bd4141',ls='--',label='Première cible de test')
    plt.xlabel('x');plt.ylabel('sin(x)');plt.legend();plt.title('Série synthétique et frontière des cibles');figure('sinus_donnees.png')
    resultats={};predictions={}
    for nom,classe in [('rnn',SimpleRNN),('lstm',SimpleLSTM),('transformer_sinus',MiniTransformer)]:
        initialiser();modele=classe();device=torch.device('cpu')
        config={'tache':'regression','epoques':300,'lr':.01,'checkpoint_toutes':10,
            'donnees':'sin(linspace(0,20,200))','fenetre':10,'train':150,'test':40,'graine':42,'batch':150}
        r=entrainer(nom,modele,donnees.chargeur(x[:150],y[:150],150),None,config,device)
        with torch.inference_mode():p=modele(x[150:]).numpy().ravel()
        r={**r,'metriques_test':metriques(y[150:].numpy().ravel(),p),
            'rechargement':rechargement(modele,classe,(x[150:],),r['poids'],device)}
        resultats[nom]=r;predictions[nom]=p.tolist()
    plt.figure(figsize=(8,4))
    for nom,r in resultats.items():plt.plot(r['histoire']['train'],label=nom)
    plt.yscale('log');plt.xlabel('Époque');plt.ylabel('MSE entraînement');plt.legend();plt.grid(alpha=.2);figure('sinus_pertes.png')
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for axe,(nom,p) in zip(axes,predictions.items()):
        axe.plot(periode[160:],y[150:].numpy().ravel(),label='Réel',color='black')
        axe.plot(periode[160:],p,'--',label=nom);axe.set_ylabel('Valeur');axe.legend();axe.grid(alpha=.2)
    axes[-1].set_xlabel('x');figure('sinus_predictions.png')
    return {'formes':{'X':list(x.shape),'y':list(y.shape),'train':150,'test':40},'modeles':resultats,
        'valeurs_test':y[150:].flatten().tolist(),'predictions':predictions,
        'evaluation':'Un pas sur fenêtres observées ; poids de la dernière époque, sans choix sur test.'}


def energie(device):
    serie,x,y,indices,masks,meta=donnees.energie()
    initialiser();modele=EnergyTransformer()
    train=donnees.chargeur(x[masks['train']],y[masks['train']],melanger=True)
    val=donnees.chargeur(x[masks['validation']],y[masks['validation']])
    config={'tache':'regression','epoques':20,'lr':.001,'graine':42,'batch':64,'fenetre':24,
        'dataset_sha256':meta['sha256'],'preparation':{'compte_min':45,'split':[.70,.15,.15],'ddof':1}}
    r=entrainer('energy_transformer',modele,train,val,config,device)
    test=donnees.chargeur(x[masks['test']],y[masks['test']])
    with torch.inference_mode():
        normalise=torch.cat([modele(lot[0].to(device)).cpu() for lot in test]).numpy().ravel()
    ix=indices[masks['test']];dates=serie.index[ix]
    observations=serie.to_numpy()[ix]
    predictions={'transformer':normalise.astype(float)*meta['ecart_type_train_kw']+meta['moyenne_train_kw'],
        'persistance':serie.to_numpy()[ix-1],'journaliere_24h':serie.to_numpy()[ix-24]}
    scores={nom:metriques(observations,p) for nom,p in predictions.items()}
    assert all(v['cibles']==len(ix) for v in scores.values())
    relecture=rechargement(modele,EnergyTransformer,(x[masks['test']][:64].to(device),),r['poids'],device)
    courbes(r['histoire'],'energie_pertes.png','Puissance horaire : sélection sur validation')
    debut=dates[0].ceil('D');grille=pd.date_range(debut,periods=168,freq='h')
    selection=pd.DataFrame({'observations':observations,**predictions},index=dates).reindex(grille)
    plt.figure(figsize=(12,5))
    for nom in selection.columns:plt.plot(selection.index,selection[nom],label=nom,linewidth=1.2)
    plt.ylabel('Puissance moyenne (kW)');plt.xlabel('Dates du test, heures locales sans fuseau déclaré')
    plt.title('Sept jours consécutifs ; les NaN interrompent les courbes');plt.legend(ncol=2);plt.grid(alpha=.2);figure('energie_sept_jours.png')
    r.update({'donnees':meta,'metriques_test_kw':scores,'rechargement':relecture,
        'graphe_sept_jours':{'debut':str(grille[0]),'fin':str(grille[-1]),'heures':168,
            'heures_sans_prediction':int(selection['transformer'].isna().sum())},
        'evaluation':'À un pas avec les observations réelles précédentes ; mêmes cibles pour les trois méthodes.',
        'transformer_ameliore_les_deux_references':all(scores['transformer'][m]<scores[b][m] for m in ['mae','rmse'] for b in ['persistance','journaliere_24h'])})
    ecrire_json(ROOT/'.cache/energie_predictions.json',{'dates':[str(d) for d in dates],'observations_kw':observations.tolist(),
        'predictions_kw':{k:p.tolist() for k,p in predictions.items()},'indices_cibles':ix.tolist()})
    del modele;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return r


def mini_gpt(device):
    x,y,xv,yv,stoi,meta=donnees.caracteres()
    itos=list(stoi)
    ecrire_json(ROOT/'poids/vocab_gpt.json',{'stoi':stoi,'itos':itos,'longueur':64})
    initialiser();modele=MiniGPT(len(stoi))
    config={'tache':'gpt','epoques':30,'lr':.003,'batch':64,'graine':42,'longueur':64,'vocabulaire':len(stoi),
        'corpus_sha256':empreinte(ROOT/'corpus.txt'),'frontiere':meta['frontiere_train'],'dropout':.1}
    r=entrainer('mini_gpt',modele,donnees.chargeur(x,y,melanger=True),donnees.chargeur(xv,yv),config,device)
    entree=xv[:2].to(device)
    modele.eval()
    futur=entree.clone();futur[:,40]=(futur[:,40]+1)%len(stoi)
    with torch.inference_mode():
        representations=modele.representations(entree);a=modele(entree);b=modele(futur)
    delta=float((a[:,:40]-b[:,:40]).abs().max());assert delta<=1e-6
    tests={'causalite':{'reussi':True,'position_modifiee':40,'positions_comparees':'0 à 39','ecart_max':delta,'tolerance':1e-6},
        'formes':{'entree':list(entree.shape),'representations':list(representations.shape),'logits':list(a.shape),'cibles':list(yv[:2].shape)}}
    tests['rechargement']=rechargement(modele,lambda:MiniGPT(len(stoi)),(entree,),r['poids'],device)
    assert json.loads((ROOT/'poids/vocab_gpt.json').read_text(encoding='utf-8'))['stoi']==stoi
    amorce='la bibliothèque';generations=[]
    with torch.inference_mode():
        for temperature in [.5,1.,1.5]:
            rng=torch.Generator(device=device).manual_seed(2026)
            ids=torch.tensor([[stoi[c] for c in amorce]],device=device)
            nouveaux=[]
            for _ in range(200):
                logits=modele(ids[:,-64:])[:,-1,:]/temperature
                suivant=torch.multinomial(torch.softmax(logits,dim=-1),1,generator=rng)
                nouveaux.append(int(suivant.item()));ids=torch.cat([ids,suivant],dim=1)
            texte=''.join(itos[i] for i in nouveaux);assert len(texte)==200
            generations.append({'temperature':temperature,'graine':2026,'amorce':amorce,'nouveaux_caracteres':len(texte),
                'texte_nouveau':texte,'texte_complet':amorce+texte})
    courbes(r['histoire'],'gpt_pertes.png','Mini-GPT : entropie croisée par caractère')
    r.update({'corpus':meta,'tests':tests,'generations':generations,'perplexite_validation_par_caractere':math.exp(r['meilleure_perte_suivie'])})
    del modele;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return r


def mini_bert(device):
    train_ids,val_ids,voc,phrases,meta=donnees.phrases_bert()
    itos=list(voc)
    ecrire_json(ROOT/'poids/vocab_bert.json',{'stoi':voc,'itos':itos,'longueur':32})
    corruption,labels,selection,stats=donnees.corrompre_mlm(val_ids,len(voc),torch.Generator().manual_seed(2026))
    padding=val_ids.eq(PAD)
    digest=__import__('hashlib').sha256(corruption.numpy().tobytes()+labels.numpy().tobytes()).hexdigest()
    train=DataLoader(TensorDataset(train_ids),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(42))
    val=DataLoader(TensorDataset(corruption,labels,padding),batch_size=64,shuffle=False,generator=torch.Generator().manual_seed(42))
    initialiser();modele=MiniBERT(len(voc))
    config={'tache':'bert','epoques':40,'lr':.001,'batch':64,'graine':42,'longueur':32,'vocabulaire':len(voc),
        'corpus_sha256':empreinte(ROOT/'corpus_bert.txt'),'validation_corruption_sha256':digest,
        'dropout':.1,'dim_feedforward':256,'selection_mlm':.15,'remplacement':[.8,.1,.1]}
    r=entrainer('mini_bert',modele,train,val,config,device)
    modele.eval()
    with torch.inference_mode():logits=torch.cat([modele(lot[0].to(device),lot[2].to(device)).cpu() for lot in val])
    exactitude=float((logits.argmax(-1)[selection]==labels[selection]).float().mean())
    remplaces=corruption.eq(MASK)&selection
    exactitude_mask=float((logits.argmax(-1)[remplaces]==labels[remplaces]).float().mean())
    original=val_ids[:4].to(device);masque=original.eq(PAD)
    modifie=original.clone();modifie[masque]=5
    with torch.inference_mode():
        a=modele(original,masque);b=modele(modifie,masque)
        representations=modele.representations(original,masque)
    delta=float((a[~masque]-b[~masque]).abs().max());assert delta<=1e-6
    tests={'padding':{'reussi':True,'masque_padding_inchange':True,'positions_padding_modifiees':int(masque.sum()),'ecart_max':delta,'tolerance':1e-6},
        'formes':{'entree':list(original.shape),'representations':list(representations.shape),'logits':list(a.shape),'labels':list(labels[:4].shape)},
        'rechargement':rechargement(modele,lambda:MiniBERT(len(voc)),(original,masque),r['poids'],device)}
    assert json.loads((ROOT/'poids/vocab_bert.json').read_text(encoding='utf-8'))['stoi']==voc
    exemples=[]
    for i in range(4):
        ids=val_ids[i].clone();positions=(ids>=5).nonzero().flatten();pos=int(positions[min(3,len(positions)-1)])
        cible=itos[int(ids[pos])];ids[pos]=MASK
        with torch.inference_mode():probas=torch.softmax(modele(ids[None].to(device),ids[None].eq(PAD).to(device))[0,pos],dim=-1)
        ps,js=torch.topk(probas,3)
        exemples.append({'phrase_originale':phrases[meta['indices_validation'][i]],'tokens_masques':[itos[j] for j in ids.tolist() if j!=PAD],
            'position_masquee':pos,'cible':cible,'top3':[{'mot':itos[j],'probabilite':p} for p,j in zip(ps.tolist(),js.tolist())]})
    courbes(r['histoire'],'bert_pertes.png','Mini-BERT : perte sur les positions sélectionnées')
    plt.figure(figsize=(8,4));plt.plot(np.arange(1,len(r['histoire']['accuracy_validation'])+1),r['histoire']['accuracy_validation'])
    plt.xlabel('Époque');plt.ylabel('Exactitude sur positions sélectionnées');plt.grid(alpha=.2);figure('bert_accuracy.png')
    r.update({'corpus':meta,'accuracy_validation_selection':exactitude,'accuracy_validation_remplacements_mask':exactitude_mask,
        'corruption_validation':{**stats,'graine':2026,'sha256':digest,'fixe_sur_toutes_les_epoques':True},'tests':tests,'exemples_masques':exemples})
    del modele;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    return r
