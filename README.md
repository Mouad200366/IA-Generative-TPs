# Travaux pratiques — IA Générative Multimodale

Ce dépôt regroupe les cinq travaux pratiques du module **IA Générative Multimodale**.
Il rassemble les notebooks, les explications, les résultats d'expérimentation et
les fichiers nécessaires à leur reproduction. Les sujets seront ajoutés au fur
et à mesure de la progression du module, conformément aux consignes du professeur.

Le premier TP porte sur les bases de PyTorch et l'apprentissage supervisé :
manipulation des données, construction de réseaux, entraînement, évaluation et
sauvegarde des modèles. Ces compétences constituent un préalable à l'étude des
modèles génératifs et multimodaux.

## Technologies utilisées

- **Python 3.12** et **JupyterLab** pour les notebooks et les expériences.
- **PyTorch** et **torchvision** pour les modèles, l'entraînement et MNIST.
- **CUDA** si un GPU compatible est disponible ; exécution sur CPU sinon.
- **NumPy** et **scikit-learn** pour les données Iris, leur préparation et les métriques.
- **Matplotlib** pour les graphiques et matrices de confusion.
- **nbformat** et **nbclient** pour la création, la validation et l'exécution des notebooks.

Les versions utilisées pour le TP1 sont indiquées dans
[`TP1-Bases-PyTorch/requirements.txt`](TP1-Bases-PyTorch/requirements.txt).

## Suivi des travaux pratiques

| TP | Sujet | État | Accès |
|---|---|---|---|
| TP1 | Bases de PyTorch | Terminé, exécuté et vérifié | [Notebook](TP1-Bases-PyTorch/TP1_IA_Generative.ipynb) · [Documentation](TP1-Bases-PyTorch/README.md) |
| TP2 | À définir selon les consignes | À venir | [Dossier TP2](TP2/README.md) |
| TP3 | À définir selon les consignes | À venir | [Dossier TP3](TP3/README.md) |
| TP4 | À définir selon les consignes | À venir | [Dossier TP4](TP4/README.md) |
| TP5 | À définir selon les consignes | À venir | [Dossier TP5](TP5/README.md) |

## Organisation du dépôt

```text
IA-Generative-TPs/
├── README.md
├── .gitignore
├── .gitattributes
├── TP1-Bases-PyTorch/
│   ├── TP1_IA_Generative.ipynb
│   ├── mnist_mlp.pth
│   ├── mnist_cnn.pth
│   ├── mnist_cnn_dropout.pth
│   ├── requirements.txt
│   └── README.md
├── TP2/README.md
├── TP3/README.md
├── TP4/README.md
└── TP5/README.md
```

Les 11 graphiques du TP1 sont intégrés au notebook. Aucun fichier de figure
externe n'est nécessaire pour consulter les résultats. Les supports de cours
et le notebook original du professeur restent locaux et ne sont pas publiés.

## TP1 — Bases de PyTorch

Le [notebook final exécuté](TP1-Bases-PyTorch/TP1_IA_Generative.ipynb) contient
quatre exercices, avec les réponses aux questions et les interprétations en français.

1. **Régression linéaire multiple — salaire.** Génération de 300 salariés fictifs,
   séparation 80/20 et standardisation sur le seul entraînement. Modèle
   `nn.Linear(2, 1)`, MSE et SGD avec `lr=0.05` pendant 500 époques. Évaluation
   par MAE en DH, prédiction d'un nouveau salaire et retour aux coefficients
   dans les unités d'origine.
2. **Classification Iris.** Séparation stratifiée 80/20, `StandardScaler`, MLP
   4 → 16 → ReLU → 3, entropie croisée et Adam avec `lr=0.01` pendant 200 époques.
   Comparaison à `nn.Linear(4, 3)`, matrice de confusion et prédiction d'une fleur.
3. **MLP MNIST.** Architecture `Flatten → Linear(784,128) → ReLU → Linear(128,10)`.
   Adam avec `lr=0.001` pendant cinq époques, lots de 64 au train et 256 au test.
   Accuracy, Precision, Recall et F1 macro, visualisation des erreurs et
   vérification des prédictions après sauvegarde/rechargement.
4. **CNN MNIST.** Deux convolutions à 16 et 32 canaux, ReLU et max pooling, puis
   couches linéaires 1568 → 128 → 10. Normalisation `(0.1307, 0.3081)`, Adam avec
   `lr=0.001` pendant trois époques, puis expérience avec Dropout `p=0.5` avant
   la dernière couche linéaire. Comparaison avec le MLP et sauvegarde des poids.

### Résultats réels du TP1

| Expérience | Résultat sur le jeu de test |
|---|---|
| Régression salaire | MAE : **381,64 DH** |
| Nouveau salarié : 7 ans, expertise 4/5 | Salaire prédit : **14 201,94 DH** |
| Coefficients du salaire | Ancienneté : **592,66 DH/an** ; expertise : **1 490,75 DH/niveau** ; biais : **4 090,30 DH** |
| Iris — MLP | Accuracy : **100 %** sur 30 fleurs |
| Iris — modèle linéaire | Accuracy : **90 %** sur les mêmes 30 fleurs |
| MNIST — MLP | Accuracy : **97,35 %** ; Precision macro : **97,36 %** ; Recall macro : **97,33 %** ; F1 macro : **97,33 %** |
| MNIST — CNN | Accuracy : **98,48 %** |
| MNIST — CNN avec Dropout | Accuracy : **98,93 %** |

Ces mesures proviennent d'une exécution complète sur **NVIDIA GeForce RTX 3060
Laptop GPU**, avec Python 3.12.14, PyTorch 2.9.1+cu128 et torchvision 0.24.1+cu128.
Les 26 cellules Python ont été exécutées dans l'ordre depuis un noyau neuf.
Les prédictions des MLP et CNN rechargés sont identiques sur les 10 000 images
de test. Les résultats et les graphiques sont déjà enregistrés dans le notebook.

Les graines sont fixées à 42. Des versions ou un matériel différents peuvent
néanmoins modifier les valeurs numériques. La comparaison MLP/CNN porte sur
les protocoles imposés par le TP : cinq contre trois époques et des
prétraitements différents. Elle n'isole donc pas l'effet de la seule architecture.

## Installation et exécution

Pour consulter le travail, ouvrir le notebook sur GitHub, dans JupyterLab ou
dans VS Code : aucun entraînement supplémentaire n'est nécessaire.

Pour reproduire les expériences, utiliser **Python 3.12 64 bits**. Depuis la
racine du dépôt, dans PowerShell :

```powershell
py -3.12 -m venv .venv
```

Installer les roues PyTorch correspondant au matériel. Pour un GPU NVIDIA
et un pilote compatible CUDA 12.8 :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Pour une installation CPU, utiliser à la place :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

Installer les autres dépendances et lancer JupyterLab :

```powershell
.\.venv\Scripts\python.exe -m pip install -r TP1-Bases-PyTorch\requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .\.venv --name tp-ia-generative --display-name "Python (TP IA Générative)"
.\.venv\Scripts\python.exe -m jupyterlab TP1-Bases-PyTorch\TP1_IA_Generative.ipynb
```

Choisir le noyau **Python (TP IA Générative)**. Pour une nouvelle exécution,
utiliser **Restart Kernel and Run All Cells**, avec `TP1-Bases-PyTorch` comme
répertoire courant du notebook.

MNIST sera téléchargé au premier lancement dans le sous-dossier `data/MNIST/`
du TP1 si nécessaire. Un accès Internet est alors requis. Le jeu complet comprend
60 000 images d'entraînement et 10 000 images de test. Iris est fourni avec
scikit-learn. Une nouvelle exécution réentraîne les modèles et réécrit leurs
poids ; elle crée également les figures PNG et un fichier de résultats JSON.

La [documentation détaillée du TP1](TP1-Bases-PyTorch/README.md) précise le
rechargement des modèles et l'interprétation des métriques. Les futurs TPs
auront leur propre documentation et leurs dépendances dans leur dossier.
