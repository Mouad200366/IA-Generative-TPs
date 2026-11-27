# TP4 — Architecture Transformer : RNN, LSTM, GPT et BERT

**Hachoum Mouad** · INTELLIGENCE ARTIFICIELLE & SCIENCES DES DONNÉES

2026-2027 · IA Générative Multimodale · **Pr. Mohamed Youssfi**

[Notebook exécuté](TP4_Architecture_Transformer.ipynb) · [Résultats détaillés](resultats_tp4.json) · [Sources](SOURCES.md)

Le TP reprend les architectures et les trois exercices du support officiel
`architecture_transformer.ipynb`. La première partie compare RNN, LSTM et Transformer
sur une sinusoïde et présente deux inférences préentraînées. La seconde traite la
puissance électrique horaire, la génération par caractères et le langage masqué.
Les onze questions de réflexion sont traitées dans le notebook.

## Résultats obtenus

Les trois modèles sinus utilisent les mêmes 150 fenêtres d'entraînement et 40 de test,
une fenêtre de 10 valeurs, Adam à 0,01 et 300 époques, sur CPU.

| Modèle | Paramètres | MSE test | MAE test | RMSE test | Entraînement |
|---|---:|---:|---:|---:|---:|
| SimpleRNN | 321 | 0,000027630 | 0,004562 | 0,005256 | 0,94 s |
| SimpleLSTM | 1 233 | 0,000002502 | 0,001355 | 0,001582 | 0,96 s |
| MiniTransformer | 2 433 | 0,000005518 | 0,002041 | 0,002349 | 2,03 s |

Le LSTM obtient la plus faible erreur dans cette expérience ; le Transformer est
plus coûteux sur ce petit exemple. Un seul découpage et une seule graine ne permettent
pas de conclure à une supériorité générale.

| Exercice | Paramètres | Protocole | Résultat |
|---|---:|---|---|
| EnergyTransformer | 26 273 | 20 époques, minimum de validation à l'époque 17 | MAE test **0,339984 kW**, RMSE **0,487630 kW** |
| Mini-GPT | 109 353 | 30 époques, minimum de validation à l'époque 30 | Entropie croisée **1,147713** par caractère ; perplexité **3,150978** |
| Mini-BERT | 168 838 | 40 époques, minimum de validation à l'époque 40 | Entropie croisée **2,159667** ; exactitude sélectionnée **54,17 %** |

Pour l'énergie, les mesures UCI sont agrégées en puissance moyenne horaire, avec au
moins 45 minutes valides par heure. Les heures insuffisantes restent NaN dans le
calendrier ; les fenêtres incomplètes sont exclues. Découpage chronologique 70/15/15,
normalisation train seulement (écart-type ddof=1), fenêtres de 24 heures,
23 805 / 5 054 / 4 932 cibles train/validation/test.

| Méthode — mêmes 4 932 cibles de test | MAE (kW) | RMSE (kW) |
|---|---:|---:|
| EnergyTransformer | 0,339984 | 0,487630 |
| Persistance, dernière heure observée | 0,389198 | 0,588385 |
| Observation 24 heures auparavant | 0,513277 | 0,763876 |

Les deux références sont améliorées sur ces deux métriques. L'évaluation utilise
les observations réelles précédentes, à un pas, sans prévision récursive. La MSE du
rapport est en **kW²**, les MAE et RMSE en **kW**. Le graphique du 25 avril au 1er mai
2010 couvre sept jours consécutifs sans heure manquante dans cette sélection ; le
code conserve les interruptions lorsqu'elles existent. L'entraînement énergétique
prend 91,18 s, hors validation et préparation.

### Démonstrations préentraînées

- **GPT-2** : véritable génération de 50 nouveaux tokens depuis
  `Artificial intelligence can help people`, température 0,8, top-p 0,9,
  échantillonnage activé, padding EOS et masque d'attention explicite.
- **BERT multilingue** : `capitale` arrive en tête pour la phrase exacte du code du
  professeur, `Rabat est la [MASK] Maroc.` (42,90 %), et pour la variante complète,
  `Rabat est la [MASK] du Maroc.` (86,06 %). Les cinq propositions, leurs scores et
  les identifiants tokenisés sont enregistrés. Un token WordPiece peut être un sous-mot.

Aucun poids préentraîné n'a été mis à jour. Les scores softmax ne constituent pas
des probabilités de vérité ; la continuation GPT-2 reste un texte échantillonné.
Les modèles sont déchargés après l'inférence ; le rapport conserve la petite
allocation GPU résiduelle réellement mesurée.

## Reproduction locale

Exécuter les commandes depuis `TP4`, avec **UV et Python 3.11** :

```powershell
uv sync --locked
uv run python creer_corpus.py
uv run python preparer_ressources.py
uv run python executer_notebook.py
uv run python finaliser_livrables.py
```

La préparation réutilise les ressources déjà présentes et ne télécharge que les
ressources manquantes : dataset officiel UCI et deux révisions Hugging Face fixées
dans le code. Aucun service payant n'est nécessaire. Le support original du professeur
n'est pas requis pour exécuter les expériences : sa référence et son empreinte sont
conservées, sans distribuer le fichier.

Le notebook est déjà exécuté. `executer_notebook.py` utilise un noyau neuf appartenant
à `.venv`, sauvegarde chaque cellule terminée et produit `verification_execution.json`.
L'exécution complète permet de reconstruire les résultats. Un entraînement terminé
est réutilisé si son rapport local `.cache/*_entrainement.json`, ses poids et sa
configuration correspondent. Sinon, un checkpoint compatible reprend à l'époque
suivante avec l'optimiseur et les états aléatoires. Une configuration incompatible
provoque une erreur au lieu d'un réentraînement implicite.

Dans une copie publique sans cache, la première exécution entraîne les six petits
modèles. Pour seulement consulter les résultats, ouvrir le notebook fourni ; les
poids et les vocabulaires sont dans `poids/`. Ne pas supprimer les caches locaux
pendant une reprise. `preparer_notebook.py` reconstruit une version **sans sorties** :
son usage est facultatif et remplace le notebook exécuté ; il doit être suivi de
l'exécution puis de la finalisation.

Exécution de référence : Python 3.11.9, PyTorch 2.9.1+cu128, CUDA 12.8,
RTX 3060 Laptop 6 Go, pilote 610.60, Transformers 5.24.1. Graine 42, float32,
deux threads CPU, algorithmes déterministes, TF32 désactivé et attention SDPA
mathématique. Les exercices et les inférences utilisent CUDA ; les trois petits
modèles sinus utilisent le CPU. Un CPU est accepté en l'absence de GPU, avec des
durées et éventuellement de faibles différences numériques propres au matériel.
Les versions complètes et les hyperparamètres figurent dans le rapport JSON.

## Validation et limites

**13 tests automatisés réussis**, plus les contrôles sur les modèles réellement
entraînés et le rechargement des six fichiers de poids, identique à l'original.
Les tests couvrent notamment les formes, les cibles décalées, le seuil horaire,
les découpages, la normalisation, le vocabulaire construit sur train et le MLM.

- Mini-GPT en `eval()` : modifier le caractère futur 40 laisse les logits 0–39
  inchangés, écart maximal **0**, tolérance absolue **10⁻⁶**.
- Mini-BERT en `eval()` : modifier 77 identifiants de padding en conservant le même
  masque laisse les logits utiles inchangés, écart maximal **0**, même tolérance.
- Trois générations mini-GPT contiennent chacune exactement **200 nouveaux
  caractères**, aux températures 0,5 / 1 / 1,5, amorce identique et graine 2026.

Le corpus GPT contient 16 197 caractères, un alphabet train de 41 caractères et un
découpage 90/10 sans fenêtre traversant la frontière. Les générations restent
fragmentaires et parfois incorrectes. Le corpus BERT comporte 446 phrases originales,
séparées en 356/90 avant construction du vocabulaire de 518 tokens. La validation
contient 38 mots inconnus sur 954 (3,98 %) ; les UNK sont exclus de la sélection MLM.
L'exactitude de 54,17 % porte sur **144 positions sélectionnées**, y compris celles
laissées inchangées. Sur les seules positions remplacées par MASK, elle vaut **51,69 %**.
La corruption de validation est fixe (graine 2026).

Les corpus contiennent des patrons récurrents et restent limités à quelques domaines.
La validation sert à choisir les poids des deux modèles de langage ; elle ne constitue
pas un test indépendant. Leurs pertes n'ont pas les mêmes unités et ne sont pas
comparables directement. L'énergie concerne un seul foyer et un découpage historique.
La fenêtre de 168 heures et les variables calendaires sont des pistes non expérimentées.

## Fichiers

`modeles.py`, `donnees.py`, `entrainement.py` et `experiences.py` regroupent les
architectures, la préparation, la reprise et les expériences ; `preentraines.py`
contient les deux inférences. `corpus_origine.json` décrit l'origine et les empreintes
des corpus originaux, rédigés avec assistance de Codex. Les huit figures sont produites
par les expériences. Aucun schéma ou extrait documentaire du support n'est reproduit.

`finaliser_livrables.py` lance les tests (`python -m pytest -q`), vérifie les livrables,
ajoute les interprétations aux cellules
Markdown sans changer leurs sorties, actualise les preuves et crée l'archive locale
`.cache/TP4_verification_ChatGPT.zip`. Elle contient le notebook, le code, les tests,
les corpus, les configurations, la documentation, les rapports, les figures et les
six petits fichiers de poids (environ 1,3 Mo au total), avec un manifeste SHA-256.
Elle exclut datasets, modèles préentraînés, support, environnement et caches.

Le [README principal](../README.md) présente le TP4 et ses résultats définitifs.
Les rapports d'exécution complets et l'archive de vérification sont conservés localement.
Les datasets, modèles préentraînés, environnements, caches, checkpoints intermédiaires
et supports originaux sont exclus de Git. Les autres TPs restent inchangés.
