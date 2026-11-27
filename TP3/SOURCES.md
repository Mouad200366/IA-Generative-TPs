# TP3 — Sources et références

## Support pédagogique

**Pr. Mohamed Youssfi**, `rag_multimodal_agentique.ipynb`, support officiel lu
intégralement avant l’implémentation (121 cellules). Le notebook original reste
dans `supports/`, exclu du suivi Git.

| Éléments du support | Cellules, indices à partir de 0 | Utilisation |
|---|---|---|
| OpenCLIP et espace texte/image commun | 6–7, 18–31 | ViT-B-32, laion2b_s34b_b79k, 512 dimensions, normalisation L2 |
| Extraction PyMuPDF | 9–15 | Texte par page, texte complet et images intégrées ; ajout du rendu des figures composées/vectorielles |
| Chunks et index Chroma | 21–41 | Fenêtres exactes de 60 tokens, chevauchement de 10, upserts et métadonnées de provenance |
| Recherche multimodale | 43–53 | Quatre directions de recherche ; ajout de la fusion texte/image et des contrôles d’entrée |
| Extensions à compléter | 54–55 | LLM local, Streamlit et séparation Agent/RAG avec MCP |
| Génération et exemples agentiques | 57–120 | Contexte multimodal, outil `consulter_pdf`, citations et boucle documentaire |

L’implémentation conserve les noms pédagogiques utiles et le modèle d’embedding
imposé. Elle remplace l’API distante illustrée dans le support par Ollama local,
ajoute un compteur d’outils strict, des références stables, une validation des
citations et une véritable communication MCP stdio. Le code commun est regroupé
dans `src/tp3rag` pour éviter sa duplication entre les interfaces.

## Article scientifique et corpus évalué

**Vaswani, A. et al. (2017).** *Attention Is All You Need*.
[Référence arXiv](https://arxiv.org/abs/1706.03762),
[révision v1](https://arxiv.org/abs/1706.03762v1).

Le corpus évalué est le PDF fourni de **11 pages**, identifié par SHA-256 :
`3c5f5a2cb399bc8fdff13f5d6da8987e185e27695307360de52bc4ea3c8e5af3`.
Les figures 1 et 2 se trouvent aux pages PDF 3 et 4. Les résultats BLEU de cette
édition sont 28,4 pour anglais-allemand et **41,0** pour anglais-français
(pages 1 et 8), et non 41,8 comme dans certaines révisions ultérieures.
Les annotations sont fondées sur ce fichier ; elles ne sont pas transférées
silencieusement à un PDF ayant une autre empreinte.

La commande de téléchargement fixe arXiv v1. Son fichier peut différer du PDF
fourni, notamment par la pagination ou des annexes. Dans ce cas, le pipeline
reste utilisable mais les annotations doivent être revérifiées avant de mesurer
Hit@k. Le PDF et les fichiers extraits ne sont pas versionnés. Les aperçus des
figures restent disponibles dans la copie locale complète, avec leur attribution.

## Réutilisation des extraits et figures

Vérification du 27 novembre 2026 : la [page arXiv v1](https://arxiv.org/abs/1706.03762v1)
renvoie à la [licence non exclusive accordée à arXiv](https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html).
Cette licence n'accorde pas directement une permission de republication aux tiers.
La [notice officielle de réutilisation](https://info.arxiv.org/help/license/reuse.html)
renvoie au titulaire des droits pour les permissions applicables.

Aucune autorisation distincte de republication des figures et des passages extraits
n'a été fournie pour ce TP3. Leur attribution est conservée, mais ne vaut pas
autorisation. L'édition publique retire les figures, les passages récupérés et
les réponses documentaires des sorties et rapports, y compris les anciennes
évaluations. Le code, les résultats numériques et la provenance sont conservés.
Les copies complètes restent locales. Le PDF complet, les extractions et le
support original demeurent exclus de Git. Voir [PUBLICATION.md](PUBLICATION.md).

## Références techniques

- [PyTorch et torchvision — paires de versions et CUDA](https://pytorch.org/get-started/previous-versions/)
- [PyMuPDF — documentation](https://pymupdf.readthedocs.io/)
- [OpenCLIP — implémentation et modèles préentraînés](https://github.com/mlfoundations/open_clip)
- [ChromaDB — documentation](https://docs.trychroma.com/)
- [LangGraph — workflows et agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [Ollama — appels d’outils](https://docs.ollama.com/capabilities/tool-calling)
- [Qwen3-VL 4B Instruct — modèle d’origine](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- [Streamlit — tests d’applications](https://docs.streamlit.io/develop/api-reference/app-testing)
- [MCP — SDK Python](https://github.com/modelcontextprotocol/python-sdk)
- [UV — gestion du projet](https://docs.astral.sh/uv/)

Intégration, rédaction et validation préparées pour **Hachoum Mouad** avec
l’assistance de **Codex**. Les architectures, le support pédagogique et les figures
restent attribués à leurs sources. Aucun code de MAINTiDOC n’est repris dans ce TP.
