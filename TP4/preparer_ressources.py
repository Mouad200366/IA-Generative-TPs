"""Préparer une fois le dataset UCI et les deux modèles préentraînés imposés."""
import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
URL_UCI = 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip'
REVISIONS = {
    'gpt2': '607a30d783dfa663caf39e06633721c8d4cfcd7e',
    'bert-base-multilingual-cased': '3f076fdb1ab68d5b2880cb87a0886f315b8146f8',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for bloc in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloc)
    return h.hexdigest()


def preparer():
    DATA.mkdir(exist_ok=True)
    fichier = DATA / 'household_power_consumption.txt'
    if not fichier.exists():
        candidats = [Path.home() / 'Downloads' / fichier.name, ROOT / 'supports' / fichier.name]
        existant = next((p for p in candidats if p.is_file()), None)
        if existant:
            shutil.copyfile(existant, fichier)
        else:
            archive = DATA / 'household_power_consumption.zip'
            if not archive.exists():
                print('Téléchargement UCI, environ 20 Mo.', flush=True)
                temporaire = archive.with_suffix('.part')
                with urllib.request.urlopen(URL_UCI, timeout=60) as source, temporaire.open('wb') as cible:
                    shutil.copyfileobj(source, cible)
                temporaire.replace(archive)
            with zipfile.ZipFile(archive) as z:
                assert fichier.name in z.namelist()
                with z.open(fichier.name) as source, fichier.open('wb') as cible:
                    shutil.copyfileobj(source, cible)
    manifeste_path = DATA / 'ressources.json'
    manifeste = json.loads(manifeste_path.read_text(encoding='utf-8')) if manifeste_path.exists() else {}
    manifeste['uci'] = {'source': URL_UCI, 'fichier': fichier.relative_to(ROOT).as_posix(),
        'sha256': sha(fichier), 'octets': fichier.stat().st_size, 'licence': 'CC BY 4.0'}
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import snapshot_download
    cache = DATA / 'hf_cache'
    for alias, repo in [('gpt2', 'openai-community/gpt2'), ('bert-base-multilingual-cased', 'google-bert/bert-base-multilingual-cased')]:
        ancien = manifeste.get(alias)
        if ancien and (ROOT / ancien['dossier'] / 'model.safetensors').is_file():
            if ancien['revision'] != REVISIONS[alias]:
                raise ValueError('Révision locale différente du protocole documenté : ' + alias)
            print(alias, ': poids déjà présents, réutilisation.', flush=True)
            continue
        # Les caches existants sont interrogés sans réseau avant tout téléchargement.
        dossier = None
        for base in [cache, Path.home() / '.cache/huggingface/hub']:
            for nom in [repo, alias]:
                try:
                    dossier = Path(snapshot_download(nom, revision=REVISIONS[alias], cache_dir=base, local_files_only=True))
                    if (dossier / 'model.safetensors').is_file():
                        break
                    dossier = None
                except Exception:
                    dossier = None
            if dossier:
                break
        if dossier is None:
            revision = REVISIONS[alias]
            print('Téléchargement nécessaire :', alias, 'révision', revision, flush=True)
            dossier = Path(snapshot_download(repo, revision=revision, cache_dir=cache,
                allow_patterns=['config.json', 'generation_config.json', 'model.safetensors',
                    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
                    'vocab.json', 'vocab.txt', 'merges.txt'], max_workers=3))
        # En cas de cache partagé extérieur, conserver un chemin local relatif.
        if not dossier.is_relative_to(ROOT):
            local = DATA / 'modeles' / alias
            local.mkdir(parents=True, exist_ok=True)
            for p in dossier.iterdir():
                if p.is_file():
                    destination = local / p.name
                    if not destination.exists():
                        try:
                            os.link(p.resolve(), destination)
                        except OSError:
                            shutil.copyfile(p, destination)
            revision = dossier.name
            dossier = local
        else:
            revision = dossier.name
        manifeste[alias] = {'repo': repo, 'revision': revision,
            'dossier': dossier.relative_to(ROOT).as_posix(),
            'poids_octets': (dossier / 'model.safetensors').stat().st_size}
        manifeste_path.write_text(json.dumps(manifeste, indent=2, ensure_ascii=False), encoding='utf-8')
    manifeste_path.write_text(json.dumps(manifeste, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Ressources prêtes, sans API payante.', flush=True)
    return manifeste


if __name__ == '__main__':
    preparer()
