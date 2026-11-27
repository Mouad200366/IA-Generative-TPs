"""Interface Streamlit ; toute la logique RAG est partagée avec le notebook/MCP."""
import asyncio
import hashlib

import streamlit as st

from tp3rag.agent import agent_direct
from tp3rag.config import Settings, lire_json
from tp3rag.retrieval import RagStore
from tp3rag.mcp_transport import ouvrir_client

st.set_page_config(page_title="TP3 · RAG multimodal", page_icon="📖", layout="wide")
st.markdown("""<style>
.block-container {max-width:1180px;padding-top:2rem}
[data-testid="stSidebar"] {background:#f0f4f8}
h1 {letter-spacing:-.04em;color:#173450}
[data-testid="stMetric"] {background:#f6f8fb;border:1px solid #e1e8ef;border-radius:12px;padding:16px}
.academic {color:#53697d;font-size:.93rem;margin-bottom:1rem}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def charger_store():
    return RagStore(Settings())


def afficher_sources(sortie, store):
    with st.expander(f"Sources consultées · {len(sortie['sources'])}", expanded=False):
        for source in sortie["sources"]:
            meta = source["metadata"]
            st.markdown(f"**{meta['pdf']} · page {meta['page']} · {meta['type']}** — `{source['reference']}`")
            st.caption(f"Similarité cosinus : {source['similarite_cosinus']:.3f} · {source['classement']}")
            if meta["type"] == "image":
                try:
                    st.image(str(store.image_source(source["id"])), width=400)
                except (ValueError, OSError):
                    st.warning("Cette source visuelle n’est plus disponible ; réindexer le PDF.")
            else:
                st.write(source["contenu"])
    with st.expander(f"Appels d’outils · {sortie['nombre_appels']}/3"):
        st.json(sortie["appels_outils"])


st.title("RAG multimodal & agentique")
st.markdown('<div class="academic">TP3 · IA Générative Multimodale · Hachoum Mouad · 2026–2027<br>Pr. Mohamed Youssfi</div>', unsafe_allow_html=True)
st.write("Interroger un PDF à partir d’une question, d’une image ou des deux. Les réponses s’appuient sur les pages retrouvées.")

try:
    store = charger_store()
except Exception:
    st.error("Impossible d’ouvrir l’index local. Vérifier l’installation et le dossier de données.")
    st.stop()

with st.sidebar:
    st.header("Document")
    upload = st.file_uploader("Charger un PDF", type=["pdf"])
    if st.button("Indexer le PDF", disabled=upload is None):
        try:
            contenu = upload.getvalue()
            if not contenu.startswith(b"%PDF-") or len(contenu) > 30 * 1024 * 1024:
                raise ValueError("PDF invalide ou supérieur à 30 Mo.")
            cible = store.settings.data / "pdfs" / (hashlib.sha256(contenu).hexdigest() + ".pdf")
            cible.parent.mkdir(exist_ok=True)
            cible.write_bytes(contenu)
            with st.spinner("Extraction et indexation OpenCLIP…"):
                bilan = store.indexer(cible, alias=upload.name)
            st.success(f"{bilan['chunks']} fragments et {bilan['images']} images disponibles.")
        except Exception as exc:
            st.error(f"Indexation interrompue ({type(exc).__name__}). Vérifier le PDF et les poids OpenCLIP.")
    documents = lire_json(store.manifest_path, {})
    choix = st.selectbox("PDF indexé", list(documents), index=0 if documents else None,
                         placeholder="Indexer un document pour commencer")
    type_resultat = st.selectbox("Sources recherchées", ["tous", "texte", "image"])
    k = st.slider("Nombre de résultats", min_value=1, max_value=12, value=6)
    image = st.file_uploader("Image de requête (facultative)", type=["png", "jpg", "jpeg", "webp"])
    transport = st.radio("Accès à la recherche", ["Direct", "MCP stdio"])
    st.caption("Le score cosinus mesure une proximité vectorielle ; il n’est pas une probabilité de vérité.")
    if st.button("Effacer la conversation"):
        st.session_state.messages = []

if choix:
    bilan = documents[choix]
    colonnes = st.columns(3)
    colonnes[0].metric("Pages PDF", bilan["pages"])
    colonnes[1].metric("Fragments", bilan["chunks"])
    colonnes[2].metric("Images indexées", bilan["images"])
else:
    st.info("Charger un PDF, ou préparer le corpus avec `uv run tp3 prepare --pdf chemin/vers/article.pdf`.")

st.session_state.setdefault("messages", [])
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sortie" in message:
            afficher_sources(message["sortie"], store)

question = st.chat_input("Poser une question sur le document…", disabled=not choix)
lancer_image = st.button("Rechercher avec l’image seule", disabled=not choix or image is None)
if question or lancer_image:
    demande = "" if lancer_image else question
    affichage = "Recherche avec l’image seule" if lancer_image else question
    st.session_state.messages.append({"role": "user", "content": affichage})
    with st.chat_message("user"):
        st.write(affichage)
    with st.chat_message("assistant"):
        try:
            image_id = store.register_image(image.getvalue()) if image else None
            with st.spinner("Recherche documentaire et réponse locale…"):
                if transport == "MCP stdio":
                    async def interroger_mcp():
                        async with ouvrir_client(store.settings, documents[choix]["document_id"]) as client:
                            return await client["agent"].aask(demande, image_id, type_resultat, k)
                    sortie = asyncio.run(interroger_mcp())
                else:
                    sortie = agent_direct(store, documents[choix]["document_id"]).ask(demande, image_id, type_resultat, k)
            st.markdown(sortie["reponse"])
            if not sortie["citations_valides"]:
                st.warning("Les références générées nécessitent une vérification dans les sources affichées.")
            afficher_sources(sortie, store)
            st.caption(f"{sortie['duree_secondes']:.1f} s · {sortie['transport']} · modèle local {store.settings.model}")
            st.session_state.messages.append({"role": "assistant", "content": sortie["reponse"], "sortie": sortie})
        except Exception as exc:
            st.error(f"La requête n’a pas abouti ({type(exc).__name__}). Vérifier Ollama et les ressources locales.")
