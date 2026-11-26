# TP2 — Sources et références

## Support pédagogique

**Pr. Mohamed Youssfi**, *Modèles génératifs avec PyTorch*, notebook officiel
`Modeles_generatifs_pytorch_avec_explications.ipynb`, cellules 2 à 55
(indices commençant à 0). Le fichier original est conservé localement.

Les architectures et les hyperparamètres suivent ce support. Les éléments
directement repris ou adaptés sont les suivants :

| Élément | Cellules du support | Utilisation dans le rendu |
|---|---|---|
| MNIST et visualisation des données | 5–6 | ToTensor, lots de 128 et réutilisation des données locales |
| Classifieur et évaluation | 8, 11–15 | Classe `Classifier`, entraînement et accuracy avec torcheval |
| Autoencodeur | 19, 22, 25, 28 | Classe `Autoencoder`, entraînement, reconstructions et espace latent |
| Prototype AE de classe connue | 32–33 | Fonction `generer_chiffre`, appliquée à la classe 1 du train |
| VAE et fonction de perte | 38, 41, 43 | Classe `VAE`, reparamétrisation, fonction `vae_loss` et entraînement |
| Exploration du VAE | 46, 49, 52, 55 | Reconstructions, moyennes latentes, 20 tirages et grille 15 × 15 |

## Adaptations et expérimentation

Le rendu complète le support par les pertes de test AE/VAE, les historiques
d’apprentissage, les graines aléatoires, les rapports JSON et la vérification
de sauvegarde/rechargement des trois modèles sur les 10 000 images de test.
Les pertes moyennes tiennent compte du nombre d’images de chaque lot.
Les résultats, figures et poids proviennent de l’exécution locale du TP2.

La rédaction, l’intégration et les contrôles ont été préparés pour **Hachoum Mouad**
avec l’assistance de **Codex**. Les architectures et fonctions pédagogiques restent
attribuées à leur source officielle.

## Références techniques

- [PyTorch — versions compatibles de torch et torchvision, variantes CUDA](https://pytorch.org/get-started/previous-versions/)
- [torchvision — jeu de données MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- [UV — intégration de PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
