# Provenance, comparaison et conditions de publication

## Référence officielle

Pr. Mohamed Youssfi, `Modeles_generatifs_pytorch_avec_explications.ipynb`, fourni
localement par l’étudiant. SHA-256 du fichier consulté :
`ac2c1d557dbba97c63a41cac53e1820a66aa1a76a75dbc0d143d05b9af28964a`.
Ce fichier original est exclu du suivi Git et reste local.

Le rendu utilise directement ses architectures et ses hyperparamètres :

| Élément | Cellules officielles, indices commençant à 0 | Traitement |
|---|---|---|
| MNIST / ToTensor / lots de 128 | 5–6 | Adaptation pour réutiliser MNIST localement, sans chemin absolu |
| Classifier | 8 | Classe reprise du professeur |
| CE, Adam, entraînement 3 époques | 11–12 | Boucle adaptée avec historique et moyenne pondérée |
| Accuracy directe et torcheval | 13, 15 | Calcul conjoint, avec contrôle de concordance |
| Autoencoder, dimension latente 2 | 19 | Classe reprise du professeur |
| MSE, Adam, entraînement 5 époques | 22 | Boucle adaptée ; ajout de la MSE test/pixel |
| Reconstruction et espace latent AE | 25, 28 | Visualisations adaptées et enregistrées |
| Prototype d’une classe connue | 32–33 | Fonction `generer_chiffre` reprise ; classe 1 sur le train |
| VAE et reparamétrisation | 38 | Classe reprise du professeur |
| BCE sommée + KL, beta=1 | 41 | Fonction `vae_loss` reprise du professeur |
| Adam, entraînement 20 époques | 43 | Réduction du gradient conservée ; ajout des historiques |
| Reconstructions, mu, tirages et grille 15 × 15 | 46, 49, 52, 55 | Adaptations de visualisation ; même domaine [−3,3] |

Les explications en français, l’évaluation test du VAE, les courbes, les rapports JSON,
le contrôle des graines et la sauvegarde/relecture sur tout le test complètent le support.
Ils ont été préparés pour Hachoum Mouad avec l’assistance de Codex. Les architectures
pédagogiques ne sont pas présentées comme une création personnelle originale.

## Référence d’Amine Zmarrou

[Dépôt consulté](https://github.com/AmineZmarrou/Activit-Pratique-N-2-Mod-les-g-n-ratifs),
branche `master`, révision `715692e6188e7b21a820177dc0652185a3903218`.
Notebook examiné : `Modeles_generatifs_pytorch.ipynb`.

Seuls ce notebook et son `pyproject.toml` ont été téléchargés dans un dossier
temporaire extérieur au dépôt de rendu. Son dataset `data/MNIST`, son environnement
et son historique Git n’ont pas été téléchargés.

La comparaison des sources montre que **toutes les cellules de code du périmètre
MNIST / classifieur / AE / VAE sont identiques à des cellules du support officiel**.
Les cellules de code 2, 5, 6, 8, 10–12, 14, 18, 20, 22, 24, 26–27, 31, 33–34,
36, 38, 40 et 42 du notebook de référence correspondent respectivement aux cellules
2, 5, 6, 8, 11–13, 15, 19, 22, 25, 28, 32–33, 38, 41, 43, 46, 49, 52 et 55
du notebook officiel. Le rendu reprend ces éléments directement du professeur.

Aucun code spécifique, résultat, poids, graphique ou texte propre à Amine n’est
republié. La consultation de son travail est explicitement créditée.
L’étudiant a déclaré avoir son autorisation d’utilisation comme référence.
Le dépôt consulté ne comporte pas de fichier de licence et son README est vide ;
les métadonnées GitHub indiquent également `license: null`. Une autorisation
explicite de republication de son code n’a donc pas pu être vérifiée dans ce dépôt.

## Écarts constatés et compléments

- Les architectures, lots, pertes et nombres d’époques de la référence correspondent
  aux consignes pour cette première partie ; aucune erreur de syntaxe n’a été détectée
  dans ses cellules de code utiles.
- La référence ne calcule pas les pertes test de l’AE et du VAE ; le rendu les ajoute.
- La moyenne de pertes par lot pour le classifieur et l’AE est remplacée par une
  moyenne pondérée par les effectifs pour tenir compte du dernier lot incomplet.
- La seconde fonction `evaluate_classifier` de la référence écrase la première :
  le rendu regroupe les deux contrôles dans une seule évaluation du test.
- Les graines, sauvegardes des trois modèles et preuves d’exécution sont ajoutées.
- Les sections GAN et diffusion du support et de la référence sont hors périmètre.
- Les sorties existantes du notebook de référence sont ignorées. Chaque sortie du
  rendu provient de l’exécution locale de Python 3.11 / PyTorch.
- Le fichier `pyproject.toml` de référence est consulté, sans le copier : le projet
  TP2 possède ses propres dépendances verrouillées et ne modifie pas celles du TP1.

## Validation avant publication

Le support local ne fournit pas de règle explicite sur la republication d’extraits
du cours ni sur les modalités de collaboration entre étudiants. La publication
reste en attente de la confirmation de l’étudiant concernant ces règles et la
republication des adaptations pédagogiques. Une consultation autorisée d’un travail
ne sera pas présentée comme une autorisation vérifiée de republier son code.
Cette validation concerne le rendu précis décrit ci-dessus ; aucun élément propre
au camarade n’est destiné à être republié.

## Documentation technique consultée

- [PyTorch — paires torch / torchvision et variantes CUDA](https://pytorch.org/get-started/previous-versions/)
- [UV — configuration des index PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)

Le projet utilise la paire PyTorch 2.9.1 / torchvision 0.24.1 avec les roues CUDA 12.8
sur Windows/Linux. Le test local vérifie la disponibilité CUDA et le calcul sur GPU.
Les bibliothèques conservent leurs licences propres ; aucune nouvelle licence n’est
attribuée aux extraits du cours.
