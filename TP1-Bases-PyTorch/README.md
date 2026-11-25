# TP 1 — IA Générative : bases de PyTorch

Ce travail pratique comprend les quatre exercices : régression linéaire multiple
sur les salaires, classification Iris, MLP MNIST et CNN MNIST avec variante Dropout.
Les champs d'identification académique sont à compléter dans la première cellule
Markdown du notebook.

## Contenu du rendu

- `TP1_IA_Generative.ipynb` : notebook final, avec les explications, réponses,
  résultats calculés et 11 graphiques intégrés.
- `mnist_mlp.pth` : paramètres du MLP MNIST.
- `mnist_cnn.pth` : paramètres du CNN principal sans Dropout.
- `mnist_cnn_dropout.pth` : paramètres du CNN avec Dropout p=0.5.
- `requirements.txt` : versions des dépendances Python utilisées.
- `README.md` : instructions de consultation et d'exécution.

Les données MNIST et l'environnement virtuel ne sont pas inclus dans le rendu.
Les fichiers du professeur ne sont pas modifiés et ne sont pas joints.

## Consulter les résultats

Ouvrir `TP1_IA_Generative.ipynb` dans JupyterLab ou VS Code. Toutes les sorties
sont déjà enregistrées : il n'est pas nécessaire de relancer les entraînements
pour consulter le travail. La finalisation de la présentation a conservé les
26 cellules Python exécutées, leurs résultats et les 11 graphiques. Aucun
entraînement n'a été relancé pendant cette finalisation.

| Exercice | Résultat mesuré sur le jeu de test |
|---|---|
| Salaire | MAE : 381,64 DH ; prédiction (7 ans, expertise 4/5) : 14 201,94 DH |
| Iris | Accuracy MLP : 100 % ; modèle linéaire : 90 % (30 fleurs) |
| MLP MNIST | Accuracy : 97,35 % ; Precision macro : 97,36 % ; Recall macro : 97,33 % ; F1 macro : 97,33 % |
| CNN MNIST | Accuracy : 98,48 % ; variante Dropout p=0.5 : 98,93 % |

Ces résultats proviennent de l'exécution complète dans un noyau neuf, avec
Python 3.12.14, PyTorch 2.9.1+cu128 et torchvision 0.24.1+cu128, sur NVIDIA
GeForce RTX 3060 Laptop GPU. Les prédictions des MLP et CNN rechargés sont
identiques sur les 10 000 images de test. La validité nbformat a été vérifiée
après la finalisation.

## Installer et lancer le notebook

Utiliser Python 3.12 64 bits. Depuis PowerShell, dans le dossier `TP1-Bases-PyTorch`,
créer son propre environnement local :

```powershell
py -3.12 -m venv .venv
```

Pour une carte NVIDIA compatible avec les roues CUDA 12.8 :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
```

Pour une installation CPU, utiliser à la place :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
```

Installer les autres dépendances, enregistrer le noyau et ouvrir le notebook :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .\.venv --name tp-ia-generative --display-name "Python (TP IA Générative)"
.\.venv\Scripts\python.exe -m jupyterlab TP1_IA_Generative.ipynb
```

Choisir le noyau **Python (TP IA Générative)**. Pour une nouvelle exécution
complète, utiliser **Restart Kernel and Run All Cells**, avec `TP1-Bases-PyTorch` comme
répertoire courant. Le notebook utilise automatiquement CUDA si disponible,
sinon CPU. Les graines sont fixées à 42 ; des versions ou un matériel différents
peuvent néanmoins modifier les valeurs numériques.

Une nouvelle exécution télécharge MNIST dans `data/MNIST/` si nécessaire :
un accès Internet est alors requis. Elle réentraîne les modèles, réécrit les
fichiers de poids et crée `figures/` ainsi que `resultats_tp1.json`. Iris est
fourni avec scikit-learn ; les données de salaire sont générées par le notebook.

## Recharger les modèles et interpréter les mesures

Les cellules de sauvegarde/rechargement recréent la même architecture, puis
chargent le `state_dict` avec `weights_only=True` et `map_location=device`.
Le prétraitement est `ToTensor()` seul pour le MLP, et `ToTensor()` suivi de
`Normalize((0.1307,), (0.3081,))` pour les CNN. Pour les poids de la variante,
recréer `MNISTClassifier(dropout=0.5)`.

La MAE du salaire est exprimée en DH. Precision, Recall et F1 du MLP MNIST
utilisent la moyenne macro sur les dix classes. Dans les matrices de confusion,
les lignes correspondent aux vraies classes et les colonnes aux prédictions.
Les pertes test par époque servent au suivi, sans sélection d'hyperparamètres.

Le MLP et le CNN respectent des protocoles différents imposés par le TP : cinq
contre trois époques et une normalisation différente. Leur comparaison ne
mesure donc pas uniquement l'effet de l'architecture. La variante Dropout est
entraînée trois époques depuis le début, avec les mêmes paramètres initiaux
et le même ordre de lots que le CNN principal.
