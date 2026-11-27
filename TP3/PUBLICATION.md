# Édition publique du TP3

Le développement et la validation technique sont terminés. Les résultats définitifs
proviennent de l'exécution locale conservée : 11 pages, 171 fragments, 2 figures
indexées, 31 tests réussis, 13/13 recherches annotées, 11/15 contrôles lexicaux,
aucune référence finale inconnue et 12,45 secondes de réponse en moyenne.
Les limites scientifiques et le statut non validé de l'import par le sélecteur
natif du navigateur restent inchangés.

Les droits de republication des figures et extraits n'étant pas établis, cette
édition ne contient pas leurs reproductions. Les sorties concernées sont signalées
explicitement, sans présenter les retraits comme une nouvelle exécution :

- Notebook : figures de l'article, aperçu de texte extrait, passages des recherches
  et réponses documentaires retirés. Les compteurs d'exécution, sorties numériques,
  explications et deux diagrammes d'architecture originaux sont conservés.
- Rapports JSON actuels et anciens : contenus des passages, réponses et requêtes
  générées dans les traces masqués. Métriques, identifiants, scores, contrôles,
  durées, numéros de page et métadonnées de provenance conservés.
- Métadonnées et empreintes publiques actualisées. Les contrôles de réponses
  décrivent les réponses locales avant masquage ; ils ne notent pas le marqueur
  de retrait affiché dans les rapports publics.

Le [manifeste](publication.json) relie les empreintes des éditions publique et
complète et recense les retraits. Les originaux restent localement dans
`.cache/publication_complete/` et `.cache/TP3_verification_ChatGPT_complete.zip`,
exclus de Git, ainsi que les données, modèles, bases vectorielles et supports.
Le PDF et le notebook original du professeur ne sont pas publiés.

## Reproduction locale

Suivre le [README](README.md) pour préparer le PDF, installer l'environnement UV,
exécuter le notebook, lancer Streamlit et utiliser MCP. Le code exécutable est
conservé et régénère les figures, passages et réponses sur la machine locale.
Les annotations correspondent au SHA-256 du PDF fourni ; elles doivent être
revérifiées pour une autre édition.

Après une nouvelle exécution locale, avant de versionner ses sorties :

```powershell
uv run python tools/prepare_publication.py --apply
uv run python tools/prepare_publication.py
```

Cette préparation sauvegarde la copie complète, retire les reproductions et
vérifie les empreintes publiques. Elle ne relance aucune expérience. Les
références au Pr. Mohamed Youssfi, à Vaswani et al. et aux outils utilisés sont
conservées dans [SOURCES.md](SOURCES.md).
