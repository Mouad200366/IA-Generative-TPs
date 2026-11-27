"""Service RAG MCP local : stdout est réservé au protocole stdio."""
from mcp.server.fastmcp import FastMCP
from tp3rag.config import Settings
from tp3rag.retrieval import RagStore

mcp = FastMCP("TP3 RAG multimodal")
_store = None


def store():
    global _store
    if _store is None:
        _store = RagStore(Settings())
    return _store


@mcp.tool()
def consulter_pdf(question: str, type_resultat: str = "tous", k: int = 6,
                  image_id: str | None = None, document_id: str | None = None) -> dict:
    """Retrouver textes et images de PDF indexés. Une image de requête doit être enregistrée ; aucun chemin arbitraire n’est accepté."""
    return store().recuperer_sources(question=question or None, type_resultat=type_resultat,
                                    k=k, image_id=image_id, document_id=document_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
