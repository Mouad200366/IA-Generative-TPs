"""Contrôle réel de démarrage des composants de l’interface Streamlit."""
import pytest
from streamlit.testing.v1 import AppTest

from tp3rag.config import ROOT


@pytest.mark.integration
def test_demarrage_streamlit():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "RAG multimodal & agentique"
    assert len(app.sidebar.selectbox) >= 2
    assert len(app.chat_input) == 1
    image_seule = next(b for b in app.button if b.label == 'Rechercher avec l’image seule')
    assert image_seule.disabled
