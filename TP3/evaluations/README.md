# Évaluation des réponses de l'agent

La référence initiale publique est conservée dans `reference_initiale/` : métriques,
provenance, instructions, critères et empreintes. Les contenus documentaires sont
masqués dans les rapports publics, y compris les essais intermédiaires.
La référence complète, la preuve d'exécution et les notebooks restent dans la
copie locale privée et l'archive complète de revue, exclues de Git.
Les fichiers `essai_*.json` conservent les variantes intermédiaires, y compris
leurs échecs ; ils ne sont pas fusionnés avec les réponses de la version finale.

Les mêmes 15 scénarios et les mêmes expressions régulières sont utilisés après
modification. Les quatre scénarios signalés sont exécutés en premier, puis les
onze autres servent au contrôle de régression. Les réponses, appels d'outils,
passages cités et durées sont enregistrés dans `../resultats_tp3.json`.

Deux lectures restent distinctes :

- **Contrôle lexical** : présence des motifs de `tests/cases.json`. Un échec peut
  correspondre à une omission réelle ou à une formulation conceptuellement correcte.
  Les motifs restent inchangés ; ils ne sont pas une accuracy sémantique.
- **Relecture sémantique qualitative** : comparaison de la question, de la réponse
et du contenu des sources effectivement citées dans la copie locale complète.
`relecture_semantique.json`
  indique les omissions, les affirmations insuffisamment étayées et les limites.
  Cette analyse est préparée avec Codex ; elle n'est ni une notation humaine
  indépendante ni un juge automatique dont l'exactitude serait garantie.

Une référence connue prouve sa provenance, pas l'appui de chaque affirmation.
La relecture demandée au LLM pendant l'exécution est un mécanisme d'amélioration,
pas une preuve de fidélité. Les échecs et régressions éventuels restent publiés
dans la comparaison, sans sélection d'une réponse parmi des répétitions identiques.

Les vecteurs de l'index sont comparés par SHA-256 avant et après validation.
La configuration OpenCLIP et le corpus sont inchangés. L'indexation du corpus
identique réutilise les embeddings existants ; seules les entrées de recherche
sont encodées pour exécuter les tests.
