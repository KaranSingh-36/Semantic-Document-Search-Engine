"""Tests for the text preprocessing pipeline and passage chunking."""
from src.chunking import chunk_document, split_into_passages
from src.preprocessing import (
    clean_for_display,
    clean_for_embedding,
    preprocess_for_tfidf,
    preprocess_tokens,
    query_terms,
    sentence_split,
    tokenize,
)


class TestPreprocessing:
    def test_lowercase_and_punctuation_removed(self):
        tokens = tokenize("Hello, WORLD! (Testing) ... done?")
        assert tokens == ["hello", "world", "testing", "done"]

    def test_urls_and_emails_removed(self):
        tokens = preprocess_tokens("Visit https://example.com/page?x=1 or www.test.org, mail admin@corp.com")
        assert not any("http" in t or "example" in t or "corp" in t or "www" in t for t in tokens)
        assert "visit" in tokens and "mail" in tokens

    def test_stopwords_removed_but_negations_kept(self):
        tokens = preprocess_tokens("The firewall is not working and it does not block attacks")
        assert "the" not in tokens and "is" not in tokens and "and" not in tokens
        assert tokens.count("not") == 2

    def test_lemmatization(self):
        tokens = preprocess_tokens("Hackers attacked the computers and were stealing passwords")
        assert {"hacker", "computer", "password", "attack", "steal"} <= set(tokens)

    def test_special_characters_and_numbers(self):
        tokens = preprocess_tokens("5G networks @ 100% speed — #fast & cheap; 2024")
        assert "5g" in tokens
        assert "2024" not in tokens and "100" not in tokens  # pure numbers are dropped
        assert all(t.isalnum() for t in tokens)

    def test_hyphenated_words_are_split(self):
        assert tokenize("machine-learning-based models") == ["machine", "learning", "based", "models"]

    def test_lemmatization_is_context_free(self):
        # the same word must map to the same lemma in a short query and in a sentence
        assert preprocess_tokens("supervised learning")[-1] == preprocess_tokens(
            "Supervised learning is a paradigm in machine learning.")[1]

    def test_empty_and_whitespace_text(self):
        assert preprocess_tokens("") == []
        assert preprocess_tokens("   \n\t ") == []
        assert preprocess_tokens("!!! ??? 123") == []
        assert preprocess_for_tfidf("") == ""

    def test_display_cleaning_preserves_original_wording(self):
        text = "Dr. Smith’s  study\r\n\r\n\r\nfound   “strong” results."
        cleaned = clean_for_display(text)
        assert "Dr. Smith's study" in cleaned
        assert '"strong"' in cleaned
        assert "\n\n" in cleaned and "\n\n\n" not in cleaned

    def test_embedding_cleaning_keeps_natural_language(self):
        cleaned = clean_for_embedding("How can I protect my PC? See https://x.org for help.")
        assert cleaned == "How can I protect my PC? See for help."

    def test_query_terms(self):
        assert query_terms("How can I protect my computer from hackers?") == {"protect", "computer", "hacker"}

    def test_sentence_split(self):
        assert sentence_split("Dr. Smith went home. It was late! Was it?") == [
            "Dr. Smith went home.", "It was late!", "Was it?"]


class TestChunking:
    def test_empty_text(self):
        assert split_into_passages("") == []
        assert chunk_document("d", "   ") == []

    def test_short_document_single_chunk(self):
        chunks = chunk_document("doc1", "A short document. It has two sentences.")
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "doc1::0000"
        assert chunks[0].doc_id == "doc1"

    def test_chunks_respect_max_words_and_overlap(self):
        sentences = [f"Sentence number {i} talks about topic {i} in some detail here." for i in range(60)]
        passages = split_into_passages(" ".join(sentences), max_words=50, overlap_sentences=1, min_words=5)
        assert len(passages) > 5
        assert all(len(p.split()) <= 50 for p in passages)
        # consecutive passages share exactly one boundary sentence
        for prev, nxt in zip(passages, passages[1:]):
            last_sentence = sentence_split(prev)[-1]
            assert nxt.startswith(last_sentence)

    def test_every_sentence_is_covered(self):
        sentences = [f"Fact {i} is stated clearly." for i in range(80)]
        passages = split_into_passages(" ".join(sentences), max_words=30, min_words=5)
        joined = " ".join(passages)
        assert all(s in joined for s in sentences)

    def test_headings_start_new_passage(self):
        text = ("Intro paragraph with enough words to stand alone as a passage of text for testing. " * 3
                + "\n\nHistory\n\nThe history section starts here and continues with several more words of text.")
        passages = split_into_passages(text, max_words=150, min_words=10)
        assert len(passages) == 2
        assert passages[1].startswith("History: The history section")

    def test_very_long_sentence_is_split(self):
        passages = split_into_passages(" ".join(["word"] * 400) + ".", max_words=100, min_words=5)
        assert all(len(p.split()) <= 100 for p in passages)
        assert sum(len(p.split()) for p in passages) >= 400
