"""Headless UI smoke tests for the Streamlit app (read-only against the real index).

Skipped when the project index has not been built yet
(``python -m src.indexing --build``).
"""
import pytest

from config import settings
from src.indexing import SearchIndex

APP = str(settings.PROJECT_ROOT / "app.py")
PAGES = [
    "pages/1_Document_Upload.py",
    "pages/2_Search_Engine.py",
    "pages/3_Model_Comparison.py",
    "pages/4_Evaluation.py",
    "pages/5_Document_Explorer.py",
]

pytestmark = pytest.mark.skipif(not SearchIndex.load().is_ready, reason="project index not built")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def test_home_dashboard(app):
    assert not app.exception
    labels = [m.label for m in app.metric]
    assert {"Indexed documents", "Indexed passages", "Uploaded documents", "Last index update"} <= set(labels)


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(app, page):
    app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


@pytest.mark.parametrize("mode", ["Semantic Search (Sentence-BERT)", "Traditional TF-IDF Search",
                                  "Compare Both Methods"])
def test_search_modes(app, mode):
    app.switch_page("pages/2_Search_Engine.py")
    app.run()
    app.text_input[0].set_value("How can I protect my computer from hackers?")
    app.radio[0].set_value(mode)
    app.get("form_submit_button")[0].click()
    app.run()
    assert not app.exception
    titles = [m.value for m in app.markdown if m.value.startswith("**") and "#1 ·" in m.value]
    assert titles and all("Computer security" in t for t in titles)


def test_model_comparison(app):
    app.switch_page("pages/3_Model_Comparison.py")
    app.run()
    app.text_input[0].set_value("how do plants make food from sunlight")
    app.get("form_submit_button")[0].click()
    app.run()
    assert not app.exception
    assert any(m.label == "Common documents" for m in app.metric)


def test_document_explorer_deep_link(app):
    app.switch_page("pages/5_Document_Explorer.py")
    app.query_params["doc"] = "wiki_phishing"
    app.run()
    assert not app.exception
    assert "Phishing" in [s.value for s in app.subheader]
