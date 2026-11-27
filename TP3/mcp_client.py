"""Client MCP et agent LangGraph : point d’entrée local."""
import asyncio
from tp3rag.mcp_transport import main, ouvrir_client

if __name__ == "__main__":
    asyncio.run(main())
