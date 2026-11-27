"""Démonstrations réelles GPT-2 et BERT multilingue, sans entraînement."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer

from entrainement import ROOT, ecrire_json, initialiser


def gpt2(ressource,device):
    initialiser()
    chemin=ROOT/ressource['dossier']
    tokenizer=AutoTokenizer.from_pretrained(chemin,local_files_only=True)
    modele=AutoModelForCausalLM.from_pretrained(chemin,local_files_only=True).to(device).eval()
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    modele.config.pad_token_id=tokenizer.pad_token_id
    modele.generation_config.pad_token_id=tokenizer.pad_token_id
    amorce='Artificial intelligence can help people'
    entree=tokenizer(amorce,return_tensors='pt').to(device)
    with torch.inference_mode():
        sortie=modele.generate(**entree,max_new_tokens=50,do_sample=True,temperature=.8,top_p=.9,
            pad_token_id=tokenizer.pad_token_id)
    resultat={'modele':'gpt2','repo':ressource['repo'],'revision':ressource['revision'],
        'device':str(device),'graine':42,'amorce':amorce,'input_ids':entree['input_ids'][0].tolist(),
        'tokens':tokenizer.convert_ids_to_tokens(entree['input_ids'][0].tolist()),
        'configuration_generation':{'max_new_tokens':50,'do_sample':True,'temperature':.8,'top_p':.9},
        'padding_token_id':tokenizer.pad_token_id,'nouveaux_tokens':int(sortie.shape[1]-entree['input_ids'].shape[1]),
        'texte_genere':tokenizer.decode(sortie[0],skip_special_tokens=True),'entrainement_effectue':False}
    return resultat


def bert(ressource,device):
    initialiser()
    chemin=ROOT/ressource['dossier']
    tokenizer=AutoTokenizer.from_pretrained(chemin,local_files_only=True)
    modele=AutoModelForMaskedLM.from_pretrained(chemin,local_files_only=True).to(device).eval()
    resultats=[]
    for phrase in [f'Rabat est la {tokenizer.mask_token} Maroc.',f'Rabat est la {tokenizer.mask_token} du Maroc.']:
        entree=tokenizer(phrase,return_tensors='pt').to(device)
        positions=(entree['input_ids'][0]==tokenizer.mask_token_id).nonzero().flatten()
        assert len(positions)==1
        position=int(positions[0])
        with torch.inference_mode():logits=modele(**entree).logits[0,position]
        probabilites=torch.softmax(logits,dim=-1)
        top_probs,top_ids=torch.topk(probabilites,5)
        propositions=[]
        for proba,indice in zip(top_probs.tolist(),top_ids.tolist()):
            ids=entree['input_ids'][0].tolist();ids[position]=indice
            propositions.append({'id':indice,'token':tokenizer.convert_ids_to_tokens(indice),
                'remplacement':tokenizer.decode([indice]).strip(),'probabilite':proba,
                'phrase':tokenizer.decode(ids,skip_special_tokens=True)})
        resultats.append({'phrase':phrase,'input_ids':entree['input_ids'][0].tolist(),
            'position_masque':position,'somme_probabilites':float(probabilites.sum()),'top5':propositions})
    return {'modele':'bert-base-multilingual-cased','repo':ressource['repo'],'revision':ressource['revision'],
        'device':str(device),'phrases':resultats,'entrainement_effectue':False}


def executer(device):
    ressources=json.loads((ROOT/'data/ressources.json').read_text(encoding='utf-8'))
    resultats={}
    for nom,fonction,cle in [('gpt2',gpt2,'gpt2'),('bert_multilingue',bert,'bert-base-multilingual-cased')]:
        config={'ressource':ressources[cle],'graine':42,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        p=ROOT/'.cache'/f'{nom}_inference.json'
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'))
            if r['signature']!=signature:raise ValueError('Inférence sauvegardée incompatible.')
            print(nom,': inférence réelle déjà sauvegardée, réutilisation.',flush=True)
        else:
            print('Inférence réelle :',nom,flush=True)
            r=fonction(ressources[cle],device)
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            r['memoire_gpu_apres_liberation_octets']=torch.cuda.memory_allocated() if device.type=='cuda' else 0
            r['signature']=signature;ecrire_json(p,r)
        resultats[nom]=r
    return resultats
