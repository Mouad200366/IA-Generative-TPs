"""Client MCP stdio et agent LangGraph séparé du service de recherche."""
import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tp3rag.agent import DocumentAgent
from tp3rag.config import ROOT, Settings


@asynccontextmanager
async def ouvrir_client(settings=None, document_id=None):
    settings = settings or Settings()
    # Le processus principal utilise le GPU pour CLIP. Le serveur MCP séparé
    # utilise le CPU par défaut pour éviter deux copies GPU face au VLM (6 Go).
    device_mcp = "cpu" if settings.clip_device == "auto" else settings.clip_device
    env = {**os.environ, "TP3_DATA_DIR": str(settings.data), "TP3_CLIP_DEVICE": device_mcp}
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], env=env)
    with (settings.data / "mcp_stderr.log").open("a", encoding="utf-8") as log:
        async with stdio_client(params, errlog=log) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                decouverte = await session.list_tools()
                if "consulter_pdf" not in [t.name for t in decouverte.tools]:
                    raise RuntimeError("Le serveur MCP n’expose pas consulter_pdf.")

                async def retrieval(**arguments):
                    arguments["document_id"] = document_id
                    arguments["question"] = arguments.get("question") or ""
                    retour = await session.call_tool("consulter_pdf", arguments=arguments)
                    if retour.isError:
                        raise RuntimeError("Le service MCP a rejeté la requête.")
                    if retour.structuredContent:
                        return retour.structuredContent
                    return json.loads(next(x.text for x in retour.content if x.type == "text"))

                yield {"agent": DocumentAgent(retrieval, settings, transport="mcp-stdio"),
                       "rechercher": retrieval, "outils": [t.name for t in decouverte.tools]}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default="Quelle est la contribution principale du Transformer ?")
    args = parser.parse_args()
    async with ouvrir_client() as client:
        print("Outils découverts :", client["outils"])
        resultat = await client["agent"].aask(args.question, type_resultat="texte")
        print(json.dumps(resultat, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
