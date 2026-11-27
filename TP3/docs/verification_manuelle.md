# Vérification manuelle de Streamlit

**Statut du parcours navigateur : non validé.** L'automatisation n'a pas ouvert
le sélecteur natif. AppTest a testé les composants avec les vrais octets du PDF
et d'une image, ainsi que le véritable retriever et Ollama. Cela ne valide pas
le transfert de fichiers depuis le navigateur.

Ouvrir `http://127.0.0.1:8503`. Si le serveur est arrêté, depuis TP3 :

```powershell
uv run streamlit run app.py --server.address=127.0.0.1 --server.port=8503
```

1. Dans **Charger un PDF**, choisir le PDF fourni *Attention Is All You Need*,
   puis **Indexer le PDF**. Vérifier le message de succès et sélectionner ce
   document dans **PDF indexé**. Pour ce corpus : 11 pages, 171 fragments, 2 images.
   Un import du même PDF réutilise les vecteurs existants.
2. Dans **Image de requête**, choisir une figure extraite du corpus. Une copie
   locale de la figure 1 est disponible dans `data/verification/figure1.png`. Une image
   PNG/JPEG fournie par l'utilisateur convient aussi.
3. Avec une image chargée, envoyer une question, par exemple « Explique cette
   architecture à partir du document ». Vérifier la réponse, les pages et les
   aperçus dans **Sources consultées**, puis **Appels d'outils** : l'appel doit
   contenir du texte et un `image_id`, avec un classement RRF.
4. Sans saisir de question, cliquer **Rechercher avec l'image seule**. Vérifier
   une nouvelle réponse et ses sources. Dans la trace, `question` doit être `null`
   et `image_id` renseigné : aucun texte n'est ajouté à l'embedding de recherche.
5. Pour un contrôle supplémentaire du transport, choisir **MCP stdio** et refaire
   une demande. Vérifier le transport annoncé et au plus trois appels.

Noter pour chaque étape : succès ou échec, message affiché, document sélectionné
et éventuelle anomalie. Une réponse affichée ne garantit pas sa fidélité : comparer
les affirmations aux passages cités. Ne passer le statut navigateur à « validé »
qu'après cette observation effective.
