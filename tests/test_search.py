"""Tests for TF-IDF search, Sentence-BERT search, indexing, comparison and evaluation.

Uses the real ``all-MiniLM-L6-v2`` model (loaded from the local cache) on a
small corpus written for the tests, stored in an isolated temp directory.
"""
import numpy as np
import pytest

from src import semantic_search as sbert
from src.comparison import compare_methods
from src.error_analysis import ErrorAnalyzer
from src.evaluation import evaluate, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from src.indexing import METHOD_SEMANTIC, METHOD_TFIDF, SearchIndex
from tests.conftest import make_docx, make_pdf

CORPUS = [
    ("sec", "Cybersecurity", "Computer security",
     "Computer security protects systems and networks from attackers. Firewalls block suspicious "
     "traffic, antivirus software removes malware such as viruses and trojans, and strong passwords "
     "with two-factor authentication stop intruders from taking over accounts."),
    ("phish", "Cybersecurity", "Phishing",
     "Phishing is a scam in which criminals send fraudulent emails that imitate banks to trick people "
     "into revealing login credentials. Users should verify the sender before clicking links."),
    ("climate", "Environment", "Climate change",
     "Climate change is the long-term warming of the planet caused by greenhouse gases such as carbon "
     "dioxide released when burning coal, oil and gas. It leads to rising sea levels and heatwaves."),
    ("solar", "Environment", "Solar power",
     "Solar power converts sunlight into electricity using photovoltaic panels. It is a renewable "
     "energy source that produces no emissions while operating."),
    ("diabetes", "Healthcare", "Diabetes",
     "Diabetes is a chronic disease in which blood sugar levels are too high because the pancreas does "
     "not produce enough insulin or the body cannot use it effectively."),
    ("antibiotic", "Healthcare", "Antibiotic",
     "An antibiotic is a medicine that kills bacteria or stops their growth. Penicillin was the first "
     "antibiotic. Overuse leads to antimicrobial resistance."),
    ("literacy", "Education", "Literacy",
     "Literacy is the ability to read and write. Schools teach children phonics, vocabulary and reading "
     "comprehension so that they can understand written texts."),
    ("nlp", "Artificial Intelligence", "Natural language processing",
     "Natural language processing enables computers to understand, interpret and generate human "
     "language, for example in translation systems, chatbots and speech assistants."),
]


def records():
    return [{"doc_id": i, "category": c, "title": t, "content": x, "source": "test", "file_name": f"{i}.txt"}
            for i, c, t, x in CORPUS]


@pytest.fixture
def index(isolated_storage):
    idx = SearchIndex()
    report = idx.add_documents(records())
    assert len(report.added) == len(CORPUS)
    return idx


class TestTfidf:
    def test_keyword_query_ranks_matching_document_first(self, index):
        response = index.search("phishing emails banks", METHOD_TFIDF, top_k=3)
        assert response.results[0].doc_id == "phish"
        assert response.processed_query == "phishing email bank"

    def test_scores_are_sorted_and_in_range(self, index):
        results = index.search("energy from sunlight and carbon emissions", METHOD_TFIDF, top_k=8).results
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(0 < s <= 1.0001 for s in scores)

    def test_only_documents_sharing_terms_are_returned(self, index):
        results = index.search("penicillin", METHOD_TFIDF, top_k=8).results
        assert [r.doc_id for r in results] == ["antibiotic"]

    def test_out_of_vocabulary_query(self, index):
        response = index.search("xylophone quasar", METHOD_TFIDF)
        assert response.results == []
        assert "vocabulary" in response.message

    def test_empty_query(self, index):
        assert index.search("   ", METHOD_TFIDF).message == "Please enter a search query."


class TestSemantic:
    def test_embeddings_are_real_and_normalised(self, index):
        assert index.embeddings.shape == (len(index.chunks), 384)
        assert np.allclose(np.linalg.norm(index.embeddings, axis=1), 1.0, atol=1e-4)
        assert index.embeddings.std() > 0.01  # not constant / placeholder vectors

    def test_paraphrase_without_shared_keywords(self, index):
        # None of "protect", "hacker" occurs in the corpus; TF-IDF can only use "computer".
        response = index.search("How can I protect my computer from hackers?", METHOD_SEMANTIC, top_k=3)
        assert response.results[0].doc_id == "sec"

    def test_semantic_finds_medicine_query(self, index):
        response = index.search("drugs that fight bacterial infections", METHOD_SEMANTIC, top_k=2)
        assert response.results[0].doc_id == "antibiotic"

    def test_semantic_returns_scores_for_oov_query(self, index):
        response = index.search("global warming", METHOD_SEMANTIC, top_k=1)
        assert response.results[0].doc_id == "climate"

    def test_category_filter_and_top_k(self, index):
        results = index.search("disease treatment", METHOD_SEMANTIC, top_k=5, categories=["Healthcare"]).results
        assert {r.category for r in results} == {"Healthcare"}
        assert len(index.search("anything", METHOD_SEMANTIC, top_k=3).results) == 3

    def test_max_pool_by_document(self):
        scores = np.array([0.1, 0.9, 0.3, 0.5, 0.2], dtype=np.float32)
        docs = np.array([0, 0, 2, 2, 0])
        doc_scores, best = sbert.max_pool_by_document(scores, docs, 3)
        assert doc_scores[0] == pytest.approx(0.9) and best[0] == 1
        assert doc_scores[2] == pytest.approx(0.5) and best[2] == 3
        assert doc_scores[1] == -np.inf and best[1] == -1


class TestIndexing:
    def test_persistence_roundtrip(self, index):
        expected = [r.doc_id for r in index.search("solar electricity", METHOD_SEMANTIC, 5).results]
        loaded = SearchIndex.load()
        assert loaded.is_ready
        assert [r.doc_id for r in loaded.search("solar electricity", METHOD_SEMANTIC, 5).results] == expected
        assert loaded.tfidf_model.doc_matrix.shape[0] == len(CORPUS)

    def test_incremental_add_reuses_existing_embeddings(self, index):
        before = index.embeddings.copy()
        new = {"doc_id": "volcano", "category": "Science", "title": "Volcano",
               "content": "A volcano is an opening in the Earth's crust through which molten lava, ash and "
                          "gases escape during an eruption."}
        report = index.add_documents([new])
        assert report.added == ["volcano"]
        assert np.array_equal(index.embeddings[: len(before)], before)
        assert index.search("lava eruption", METHOD_TFIDF, 1).results[0].doc_id == "volcano"
        assert index.search("mountain that spews fire", METHOD_SEMANTIC, 1).results[0].doc_id == "volcano"

    def test_duplicate_documents_are_skipped(self, index):
        duplicate = {**records()[0], "doc_id": "sec_copy"}
        report = index.add_documents([duplicate])
        assert report.added == [] and "duplicate" in report.skipped[0]

    def test_remove_document_keeps_index_aligned(self, index):
        index.remove_documents(["climate"])
        assert index.get_document("climate") is None
        assert all(c["doc_id"] != "climate" for c in index.chunks)
        assert index.embeddings.shape[0] == len(index.chunks)
        assert index.tfidf_model.doc_matrix.shape[0] == len(CORPUS) - 1
        results = index.search("greenhouse gases warming", METHOD_SEMANTIC, 8).results
        assert "climate" not in [r.doc_id for r in results]
        reloaded = SearchIndex.load()
        assert reloaded.is_ready and reloaded.get_document("climate") is None

    def test_rebuild_reproduces_rankings(self, index):
        query = "children learning to read"
        before = [r.doc_id for r in index.search(query, METHOD_SEMANTIC, 5).results]
        index.rebuild()
        assert [r.doc_id for r in index.search(query, METHOD_SEMANTIC, 5).results] == before
        assert index.embeddings.shape[0] == len(index.chunks)

    def test_upload_txt_pdf_docx_and_delete(self, index, isolated_storage):
        txt = ("Quantum computers use qubits that exploit superposition and entanglement to solve "
               "certain problems faster than classical computers.").encode()
        pdf = make_pdf(["Volcanoes erupt molten rock called lava from magma chambers beneath the crust."],
                       title="Volcano notes")
        docx = make_docx(["Photosynthesis lets plants turn sunlight, water and carbon dioxide into sugar."])
        ids = []
        for name, data in (("quantum.txt", txt), ("volcano.pdf", pdf), ("plants.docx", docx)):
            report = index.add_uploaded_file(name, data, category="Science")
            assert len(report.added) == 1, report
            ids.append(report.added[0])
        assert len(list((isolated_storage / "uploads").iterdir())) == 3
        assert index.stats()["uploaded"] == 3
        assert index.search("qubits superposition", METHOD_TFIDF, 1).results[0].doc_id == ids[0]
        assert index.search("how plants make food from light", METHOD_SEMANTIC, 1).results[0].doc_id == ids[2]
        # uploading the same file again is detected as a duplicate
        assert index.add_uploaded_file("quantum_copy.txt", txt).added == []
        index.remove_documents(ids)
        assert index.stats()["uploaded"] == 0
        assert list((isolated_storage / "uploads").iterdir()) == []

    def test_fingerprint_change_requires_rebuild(self, index, monkeypatch):
        import src.indexing as indexing
        monkeypatch.setitem(indexing.INDEX_FINGERPRINT, "chunk_max_words", 999)
        assert SearchIndex.load().needs_rebuild


class TestComparisonAndEvaluation:
    def test_compare_methods(self, index):
        cmp = compare_methods(index, "How can I protect my computer from hackers?", top_k=3)
        tf = {r.doc_id for r in cmp.tfidf.results}
        se = {r.doc_id for r in cmp.semantic.results}
        assert set(cmp.common) == tf & se
        assert set(cmp.tfidf_only) == tf - se and set(cmp.semantic_only) == se - tf
        assert 0 <= cmp.jaccard <= 1
        assert cmp.explanations
        assert len(cmp.tfidf_all_scores) == len(CORPUS)

    def test_metrics(self):
        ranking = ["a", "x", "b", "y", "z"]
        relevant = {"a", "b", "c"}
        assert precision_at_k(ranking, relevant, 3) == pytest.approx(2 / 3)
        assert recall_at_k(ranking, relevant, 5) == pytest.approx(2 / 3)
        assert reciprocal_rank(["x", "y", "a"], relevant) == pytest.approx(1 / 3)
        assert reciprocal_rank(["x", "y"], relevant) == 0.0
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)
        # swapped order: DCG = 1/1 + 3/log2(3); IDCG = 3/1 + 1/log2(3)
        expected = (1 + 3 / np.log2(3)) / (3 + 1 / np.log2(3))
        assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) == pytest.approx(expected)
        assert ndcg_at_k(["x"], {"a": 2}, 1) == 0.0

    def test_evaluate_and_error_analysis(self, index):
        eval_set = {"queries": [
            {"query_id": "T1", "query": "stop hackers breaking into my PC", "relevant": {"sec": 2, "phish": 1}},
            {"query_id": "T2", "query": "penicillin", "relevant": {"antibiotic": 2}},
            {"query_id": "T3", "query": "global warming", "relevant": {"climate": 2, "solar": 1}},
            {"query_id": "T4", "query": "learning to read", "relevant": {"literacy": 2}},
            {"query_id": "T5", "query": "missing doc", "relevant": {"not_indexed": 2}},
        ]}
        result = evaluate(index, eval_set, k_values=(1, 3))
        assert [q["query_id"] for q in result.skipped] == ["T5"]
        assert len(result.per_query) == 8  # 4 queries x 2 methods
        assert set(result.summary["method"]) == {METHOD_TFIDF, METHOD_SEMANTIC}
        for col in ("P@1", "R@3", "nDCG@3", "MRR"):
            assert result.per_query[col].between(0, 1).all()
        t2 = result.per_query.query("query_id == 'T2' and method == 'tfidf'").iloc[0]
        assert t2["P@1"] == 1.0 and t2["MRR"] == 1.0
        analysis = ErrorAnalyzer(index, result, k=3).run()
        assert set(analysis) == {"comparison_table", "irrelevant_top", "missed_relevant",
                                 "tfidf_better", "sbert_better"}
        assert len(analysis["comparison_table"]) == 4
