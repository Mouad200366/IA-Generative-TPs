# Sources et attribution — TP4

## Support pédagogique

**Pr. Mohamed Youssfi**, `architecture_transformer.ipynb`, support officiel du module
IA Générative Multimodale. Les 56 cellules ont été consultées : exemples
RNN/LSTM/Transformer (cellules 8–29), GPT-2 et BERT (31–35), puis trois exercices et
leurs questions (37–54, indices à partir de zéro).

Les architectures et protocoles imposés sont adaptés de ce support. Le code de
préparation, la reprise, les contrôles, les corpus et la présentation du rendu sont
réalisés pour Hachoum Mouad avec assistance de Codex. Les résultats proviennent des
exécutions locales enregistrées. Le support original et ses images restent locaux ;
aucune autorisation générale de redistribution de ce support n'est revendiquée.

Empreinte du support consulté :
`17dbf76ffdd201515a8bd9d20c487e53b2e352e7f5392dbdb09f14a10c8495ba`.

## Données et corpus

- **Hebrail, G. et Berard, A. (2006)**, *Individual Household Electric Power
  Consumption*, UCI Machine Learning Repository,
  [fiche officielle](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
  [DOI : 10.24432/C58K54](https://doi.org/10.24432/C58K54),
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  `Global_active_power` est une puissance active moyenne en kW. Le TP transforme les
  données minute en moyennes horaires, conserve le calendrier incomplet et normalise
  à partir de l'entraînement. Le fichier original reste exclu de Git et de l'archive.
- `corpus.txt` et `corpus_bert.txt` : textes français originaux narratifs et
  semi-synthétiques, rédigés et assemblés pour ce TP avec assistance de Codex, sans
  extrait externe. Attribution : **Hachoum Mouad, corpus pédagogiques TP4, 2026**.
  Leur origine, leurs tailles, leurs limites et leurs SHA-256 figurent dans
  `corpus_origine.json`. Aucune licence de réutilisation n'est attribuée à ces corpus.
- Les graphiques sont des tracés originaux Matplotlib issus des expériences ; ils
  ne reproduisent pas les figures du support. Les courts exemples préentraînés sont
  des sorties locales identifiées comme telles, sans garantie factuelle.

## Modèles préentraînés

| Modèle | Source officielle | Licence annoncée | Révision utilisée |
|---|---|---|---|
| GPT-2 | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | MIT | `607a30d783dfa663caf39e06633721c8d4cfcd7e` |
| BERT multilingue cased | [google-bert/bert-base-multilingual-cased](https://huggingface.co/google-bert/bert-base-multilingual-cased) | Apache 2.0 | `3f076fdb1ab68d5b2880cb87a0886f315b8146f8` |

Les poids et les fichiers téléchargés restent locaux. Le code appelle les modèles
sans les modifier ; ni ces poids ni leur cache ne sont redistribués dans le rendu.

## Références techniques

- [PyTorch](https://pytorch.org/docs/stable/) : RNN, LSTM, TransformerEncoder,
  MultiheadAttention, pertes et sauvegarde des états.
- [Distribution PyTorch](https://pytorch.org/get-started/previous-versions/) :
  version 2.9.1 avec distribution CUDA 12.8 pour l'exécution de référence.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/),
  [GPT-2](https://huggingface.co/docs/transformers/model_doc/gpt2),
  [BERT](https://huggingface.co/docs/transformers/model_doc/bert) : AutoTokenizer,
  AutoModelForCausalLM, AutoModelForMaskedLM et inférence.
- [UV](https://docs.astral.sh/uv/) : environnement Python 3.11 et verrouillage.
- [NumPy](https://numpy.org/doc/), [pandas](https://pandas.pydata.org/docs/),
  [Matplotlib](https://matplotlib.org/stable/),
  [nbclient](https://nbclient.readthedocs.io/) et
  [pytest](https://docs.pytest.org/) : préparation, figures, exécution et vérification.
