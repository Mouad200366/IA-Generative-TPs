"""Tests d’intégration avec le véritable corpus, OpenCLIP et le service MCP."""
import asyncio

import numpy as np
import pytest

from tp3rag.config import ROOT, Settings, lire_json
from tp3rag.mcp_transport import ouvrir_client
from tp3rag.retrieval import RagStore

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def corpus():
    if not (ROOT / "data" / "documents.json").is_file():
        pytest.skip("Préparer et indexer le PDF avant les tests d’intégration.")
    return RagStore(Settings())


def test_index_reel(corpus):
    r = corpus.collection.get(include=["embeddings", "metadatas"])
    assert len(r["ids"]) > 0
    assert np.asarray(r["embeddings"]).shape[1] == 512
    assert np.allclose(np.linalg.norm(r["embeddings"], axis=1), 1, atol=1e-5)
    assert {m["type"] for m in r["metadatas"]} == {"texte", "image"}


def test_figures_texte_image(corpus):
    for query, figure in [("Transformer model architecture encoder decoder diagram", 1),
        ("Scaled Dot Product Attention Multi Head Attention two diagrams", 2)]:
        r = corpus.rechercher(query, type_resultat="image", k=1)
        assert r[0]["metadata"]["figure"] == figure


def test_image_vers_image_texte_et_hybride(corpus):
    image = corpus.collection.get(where={"type": "image"}, include=["metadatas"])
    index = next(i for i, m in enumerate(image["metadatas"]) if m["figure"] == 1)
    fichier = corpus.image_source(image["ids"][index])
    ident = corpus.register_image(fichier.read_bytes())
    r = corpus.rechercher(image_id=ident, type_resultat="image", k=1)
    assert r[0]["id"] == image["ids"][index] and r[0]["similarite_cosinus"] > 0.999
    r = corpus.rechercher(image_id=ident, type_resultat="texte", k=4)
    assert len(r) == 4 and all(x["metadata"]["type"] == "texte" for x in r)
    r = corpus.rechercher("encoder decoder", image_id=ident, k=4)
    assert len(r) == 4 and all(x["classement"].startswith("RRF") for x in r)


def test_decouverte_et_appel_mcp_reels(corpus):
    async def verifier():
        async with ouvrir_client(corpus.settings) as client:
            assert "consulter_pdf" in client["outils"]
            r = await client["rechercher"](question="scaled dot product attention", type_resultat="image", k=2, image_id=None)
            assert len(r["sources"]) == 2 and len(r["images"]) == 2
            assert all(len(x["base64"]) > 100 for x in r["images"])
    asyncio.run(verifier())


def test_rapport_agent_reel():
    r = lire_json(ROOT / "resultats_tp3.json")
    if r is None:
        pytest.skip("Exécuter le notebook ou tp3 validate avant de contrôler son rapport.")
    assert len(r["cas"]) == 15
    assert r["mesures"]["agents_executes"] == 15
    assert r["mesures"]["budget_3_respecte"]
    assert r["mcp"]["statut"] == "succes"
    assert r["mcp"]["agent"]["nombre_appels"] >= 1
    assert r["mcp"]["agent"]["images_transmises_au_vlm"] >= 1
    assert r["capacites_llm"]["vision"]["succes"]
    assert r["capacites_llm"]["tool_calling"]["succes"]
