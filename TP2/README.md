# TP2 — Modèles génératifs avec PyTorch

| Informations académiques | Détails |
|---|---|
| Nom et prénom | Hachoum Mouad |
| Filière | INTELLIGENCE ARTIFICIELLE & SCIENCES DES DONNÉES |
| Établissement | EMSI Maarif |
| Année universitaire | 2026-2027 |
| Module | IA Générative Multimodale |
| Enseignant | Pr. Mohamed Youssfi |

## Objectifs et périmètre

Créer, entraîner et évaluer sur MNIST un modèle discriminatif, un autoencodeur (AE)
et un autoencodeur variationnel (VAE), puis explorer les représentations latentes et
la génération. La première partie demandée ne comprend aucun entraînement de GAN
ni de diffusion. Les données complètes sont utilisées : 60 000 images au train et
10 000 au test, `ToTensor()`, lots de 128, aucun apprentissage sur le test.

## Architectures et protocole du professeur

| Modèle | Architecture | Perte | Optimisation |
|---|---|---|---|
| Discriminatif | Flatten → 784 → 128 → ReLU → 10 | CrossEntropyLoss, logits sans Softmax | Adam, lr=0.001, 3 époques |
| AE | Encodeur 784 → 128 → 32 → 2 ; décodeur 2 → 32 → 128 → 784 | MSE moyenne par pixel | Adam, lr=0.001, 5 époques |
| VAE | Encodeur 784 → 256 → 64 ; mu/logvar de dimension 2 ; décodeur 2 → 64 → 256 → 784 | BCE sommée + KL, beta=1 | Adam, lr=0.001, 20 époques |

Les couches cachées utilisent ReLU ; les décodeurs AE/VAE se terminent par Sigmoid.
Le VAE utilise `z = mu + exp(0.5 * logvar) * epsilon`, avec epsilon normal standard.
La réduction `sum` de sa perte est conservée pour le gradient, conformément au cours.
Les pertes rapportées sont pondérées par le nombre d’images.

## Résultats réellement calculés

| Expérience | Résultat sur les 10 000 images de test |
|---|---|
| Classifieur — 3 époques | Accuracy torcheval : **96.49 %** (9649/10 000) |
| AE — 5 époques | MSE/pixel : **0.044107** |
| VAE — 20 époques | BCE/image : **137.6209** ; KL/image : **6.4156** ; total/image : **144.0365** |
| VAE — reconstruction à partir de mu | MSE/pixel : **0.037783** |

Exécution locale du 2026-11-26, sur **NVIDIA GeForce RTX 3060 Laptop GPU**,
Python **3.11.9**, torch **2.9.1+cu128**,
torchvision **0.24.1+cu128**, runtime CUDA **12.8**.
Durée des cellules de calcul : **326.4 secondes**, hors installation et démarrage du noyau.

Les **16 cellules Python** ont été exécutées dans l’ordre depuis un noyau neuf,
sans erreur, et les **9 graphiques** sont intégrés au notebook.
L’accuracy torcheval est contrôlée par un comptage direct.
Les trois modèles rechargés donnent des sorties déterministes identiques sur tout le test ;
pour le VAE, ce contrôle porte sur mu, logvar et decode(mu), afin d’isoler la sauvegarde du tirage aléatoire.
`verification_execution.json` est une preuve locale ignorée par Git, créée par le lanceur.

La MSE AE et la BCE+KL VAE ont des unités différentes. La MSE additionnelle du VAE
utilise decode(mu), tandis que sa perte probabiliste utilise un tirage par image
avec la graine 43. Les architectures et budgets d’entraînement diffèrent : les
valeurs ne constituent pas une comparaison contrôlée de la seule régularisation.
Le VAE permet de tirer des codes N(0,I), sans image d’entrée. Le prototype AE de
classe 1 est décodé depuis les codes moyens du train et ne constitue pas un tirage
probabiliste. Les générations et la grille permettent une inspection qualitative,
sans prétendre mesurer statistiquement la qualité, la diversité ou l’originalité.



## Lecture des figures obtenues

Les courbes diminuent sur les trois entraînements. Pour le VAE, le total passe de
193,8184 à 142,5683 par image, tandis que la KL augmente de 4,1269 à 6,4600 :
l’amélioration de reconstruction s’accompagne d’un coût accru d’écart au prior.
La perte test totale de 144,0365 est proche de la dernière moyenne d’entraînement,
sans permettre à elle seule de conclure à une généralisation parfaite.

Sur les dix exemples présentés, l’AE préserve bien les traits verticaux des 1 et la
boucle du 0, mais certains 7 et 4 ressemblent à des 9 après reconstruction. Les
nuages latents présentent des chevauchements, notamment entre plusieurs chiffres
à boucles. Les codes des 1 forment une région assez distincte et leur code moyen
donne un prototype vertical reconnaissable.

Le VAE reconstruit visuellement mieux certains 7 et 2 de cet échantillon, tout en
conservant des confusions et du flou sur plusieurs chiffres. Ses vingt tirages
comprennent des formes reconnaissables de 1, 2, 3, 6 et 7, ainsi que des formes
ambiguës. La grille montre des transitions graduelles entre régions de 7, de 1 et
de chiffres à boucles ; une même forme se répète dans plusieurs tuiles voisines.
Cela illustre un espace décodable continu, sans garantir la séparation des dix
classes ni une diversité uniforme. Ces observations portent sur les figures de
cette exécution et complètent les métriques, sans les remplacer.


## Installation isolée et exécution avec UV

Installer [UV](https://docs.astral.sh/uv/getting-started/installation/) si nécessaire.
Depuis la racine du dépôt, dans PowerShell :

```powershell
cd TP2
uv sync --locked
uv run python executer_notebook.py
```

`.python-version` sélectionne Python 3.11 ; `pyproject.toml` et `uv.lock` définissent
un environnement propre à TP2 dans `TP2/.venv`. Ils ne modifient pas le TP1.
UV peut télécharger Python 3.11 si absent. Le lanceur crée automatiquement un
nouveau noyau avec le Python courant, contrôle la syntaxe, exécute les cellules,
enregistre le notebook et produit le rapport local de vérification.

Pour une exécution interactive, ouvrir le notebook dans VS Code et sélectionner
`TP2/.venv/Scripts/python.exe` (Windows) ou `TP2/.venv/bin/python` (Linux/macOS).
Utiliser **Restart Kernel and Run All Cells**, avec `TP2` comme répertoire courant.
Le notebook écrit ses résultats, figures et poids ; le lanceur ajoute la vérification
de l’exécution complète. Consulter le notebook déjà exécuté ne demande aucun entraînement.

Les index UV associent explicitement torch 2.9.1 et torchvision 0.24.1 aux roues
CUDA 12.8 sous Windows/Linux. Le pilote NVIDIA local a été contrôlé et le calcul
sur la RTX 3060 testé. Sans GPU, le même code sélectionne le CPU ; l’installation
verrouillée sous Windows/Linux conserve toutefois les roues CUDA et leur taille.
La configuration macOS utilise les distributions PyPI, sans CUDA.
Voir les [paires compatibles PyTorch](https://pytorch.org/get-started/previous-versions/)
et la [configuration UV/PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/).

La recherche de MNIST utilise uniquement des chemins relatifs (`data`, `../data`,
`../../data`, `../TP1-Bases-PyTorch/data`). Le dataset existant est réutilisé si
les quatre fichiers bruts sont présents. Sinon torchvision télécharge MNIST dans
`TP2/data`. Les données, environnements, caches et rapports locaux restent ignorés.
Graines 42 et algorithmes déterministes sont activés ; une autre plateforme ou
version peut néanmoins produire des résultats différents.

## Fichiers du TP2

- [Notebook exécuté](TP2_Modeles_Generatifs.ipynb)
- [Résultats et historiques JSON](resultats_tp2.json)
- [Configuration UV](pyproject.toml), [verrou des dépendances](uv.lock), [.python-version](.python-version)
- [Lanceur et vérification depuis un noyau neuf](executer_notebook.py)
- [Poids du classifieur](poids/classifier_mnist.pth), [poids AE](poids/autoencoder_mnist.pth), [poids VAE](poids/vae_mnist.pth)
- [Figures PNG](figures/) : exemples, reconstructions, nuages latents, prototype AE,
  20 générations VAE, grille 15 × 15 et courbes d’apprentissage
- [Sources, comparaison et conditions de publication](SOURCES.md)

Les poids sont des `state_dict` sur CPU. Pour recharger un modèle, exécuter sa
définition de classe puis utiliser, par exemple :

```python
ae = Autoencoder(latent_dim=2).to(device)
ae.load_state_dict(torch.load('poids/autoencoder_mnist.pth', map_location=device, weights_only=True))
ae.eval()
```

## Sources et crédits

Les architectures et le code pédagogique proviennent de **Pr. Mohamed Youssfi**,
`Modeles_generatifs_pytorch_avec_explications.ipynb`, cellules 2 à 55 ; le fichier
original n’est pas publié. La classe du classifieur, les classes AE/VAE, la perte
VAE et la fonction de génération AE sont créditées dans les cellules concernées.

Le [travail d’Amine Zmarrou](https://github.com/AmineZmarrou/Activit-Pratique-N-2-Mod-les-g-n-ratifs/tree/715692e6188e7b21a820177dc0652185a3903218)
a été consulté avec son accord déclaré par l’étudiant. Les cellules utiles sont
identiques au support officiel ; le rendu les reprend directement du professeur.
Aucun code spécifique, résultat ni poids d’Amine n’est republié. Son dépôt ne
contient pas de licence explicite. Les modalités de collaboration et la publication
des adaptations du cours restent soumises à la validation de l’étudiant avant le push.
Voir [SOURCES.md](SOURCES.md) pour la comparaison détaillée.

Intégration, explications, exécution et contrôles préparés pour **Hachoum Mouad**
avec l’assistance de **Codex**. Les éléments pédagogiques externes ne sont pas
présentés comme une réalisation personnelle originale.
