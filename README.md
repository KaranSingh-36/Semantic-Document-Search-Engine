# Semantic Document Search Engine

An end-to-end NLP application that searches a document collection with natural-language queries and compares a **classical TF-IDF keyword baseline** with **transformer-based semantic search (Sentence-BERT, `all-MiniLM-L6-v2`)**.

*MCA AI & ML, Semester III capstone project (ETMMTI303).*

---

## 1. Introduction and problem statement

Keyword search engines only find documents that contain the words typed by the user. A query such as *"How can I protect my computer from hackers?"* should also find documents about *antivirus software*, *firewalls* or *malware*, even when they never use the word "hacker".

This project builds a search engine that:

* accepts natural-language queries;
* retrieves the most relevant **documents** and the most relevant **passage** inside each one;
* shows similarity scores and highlights the query terms;
* lets users upload their own PDF, TXT and DOCX files, which become searchable immediately;
* compares keyword search and semantic search side by side;
* measures both methods on a manually labelled test set (Precision@K, Recall@K, MRR, nDCG@K) and includes an error analysis.

### Objectives

1. Collect and clean a multi-domain public document collection.
2. Implement a preprocessing pipeline that keeps both an original and a preprocessed version of every text.
3. Implement TF-IDF + cosine similarity retrieval as the baseline.
4. Implement Sentence-BERT passage retrieval with a pretrained model on CPU.
5. Build a persistent, incrementally updatable index (add, remove, rebuild).
6. Provide a Streamlit interface for search, comparison, upload, exploration and evaluation.
7. Evaluate both methods objectively and analyse their failures.

---

## 2. System architecture

```mermaid
flowchart LR
    subgraph Sources
        W[Wikipedia articles<br/>MediaWiki API] --> DL[data_loader.py<br/>download · clean · dedupe]
        U[User uploads<br/>PDF / TXT / DOCX] --> DP[document_processor.py<br/>detect · extract · validate]
    end

    DL --> IDX
    DP --> IDX

    subgraph IDX[indexing.py - SearchIndex]
        direction TB
        PRE[preprocessing.py<br/>clean text for display & SBERT<br/>tokenise · stop-words · lemmatise for TF-IDF]
        CH[chunking.py<br/>sentence-based passages ≤150 words]
        TF[tfidf_search.py<br/>TfidfVectorizer 1-2 grams<br/>doc + passage matrices]
        SB[semantic_search.py<br/>all-MiniLM-L6-v2<br/>384-d passage embeddings]
        PRE --> TF
        CH --> TF
        CH --> SB
    end

    IDX <--> ST[(Disk storage<br/>data/processed · models/tfidf · models/embeddings)]

    Q[User query] --> IDX
    IDX --> R[Ranked documents<br/>+ best passages + scores]
    R --> UI[Streamlit UI<br/>Search · Comparison · Explorer]
    IDX --> EV[evaluation.py<br/>P@K · R@K · MRR · nDCG@K]
    EV --> EA[error_analysis.py]
    EV --> REP[reports/evaluation_results]
    EA --> UI
```

### Query flow

| Step | TF-IDF (keyword) | Sentence-BERT (semantic) |
|---|---|---|
| 1 | Query is preprocessed the same way as the documents (lower-case, tokenise, stop-words, lemmatise) | Query is lightly cleaned (URLs removed); natural wording is kept |
| 2 | Transformed with the fitted `TfidfVectorizer` | Encoded with `all-MiniLM-L6-v2` into a normalised 384-d vector |
| 3 | Cosine similarity with each **document** vector | Cosine similarity with each **passage** embedding |
| 4 | Documents ranked by score (documents sharing no term are excluded) | Document score = best passage score (max-pooling) |
| 5 | Best passage = passage with the highest TF-IDF similarity | Best passage = highest-scoring passage |

---

## 3. Technologies used

| Purpose | Library |
|---|---|
| Language | Python 3.11 (3.10 also supported by `requirements.txt`) |
| Frontend | Streamlit |
| Data processing | pandas, NumPy, SciPy |
| Classical NLP | scikit-learn (`TfidfVectorizer`), NLTK (tokeniser, stop-words, WordNet lemmatiser) |
| Transformer embeddings | sentence-transformers with a PyTorch CPU backend |
| Document extraction | pypdf, python-docx |
| Visualisation | Plotly (interactive), Matplotlib (static report charts) |
| Persistence | joblib (TF-IDF), NumPy `.npy` (embeddings), JSON (metadata) |
| Testing | pytest, Streamlit `AppTest` |

There is no database, external API or cloud service. The only network access is the one-time download of the dataset, the NLTK resources and the model.

---

## 4. Dataset

### 4.1 Source

**English Wikipedia** articles, downloaded as plain text through the official MediaWiki Action API:

```
https://en.wikipedia.org/w/api.php?action=query&prop=extracts|info&explaintext=1&redirects=1&titles=<title>
```

* License: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Each document stores its source URL (`https://en.wikipedia.org/wiki/<Title>`), page id, revision id and retrieval time in `data/raw/wikipedia/manifest.json`.
* The article list is defined in `src/data_loader.py` (`WIKIPEDIA_ARTICLES`). Re-create or refresh the snapshot with `python -m src.data_loader --download`.

### 4.2 Collection statistics (actual snapshot)

| Category | Documents |
|---|---|
| Artificial Intelligence | 11 |
| Machine Learning | 12 |
| Cybersecurity | 13 |
| Healthcare | 11 |
| Education | 10 |
| Technology | 11 |
| Environment | 11 |
| Science | 10 |
| **Total** | **89** |

* 616,645 words in total; document length ranges from **1,843 to 15,116 words** (mean 6,929).
* Splitting produced **6,375 passages**, and the TF-IDF vocabulary has **289,093** unigrams and bigrams.

### 4.3 Cleaning

* Trailing link and citation sections (*See also, References, External links, Notes…*) are removed. Only the **final run** of such sections is cut, because a real content section can share the name; for example, *Air pollution* has a "Sources" section about pollution sources.
* LaTeX residue (`{\displaystyle …}`) is removed with balanced-brace matching.
* Section headings are turned into plain lines, and the chunker uses them as passage boundaries.
* **Duplicates** are detected by page id (redirects) and by a hash of whitespace- and case-normalised content.
* **Missing or empty** documents (fewer than 50 characters) are dropped. The same checks apply to uploads.

### 4.4 Fallback sample dataset

`data/raw/sample/` holds 16 short documents (2 per category) **written for this project**. They are **not external documents**. They are used only if the Wikipedia snapshot is missing and cannot be downloaded, and the app shows a warning while they are in use.

### 4.5 User documents

PDF, TXT and DOCX files can be uploaded from the **Document Upload** page. The original files are stored in `data/uploads/` and their metadata in the index.

---

## 5. Preprocessing methodology (`src/preprocessing.py`)

Two versions of every text are kept:

| Version | Used for | Steps |
|---|---|---|
| **Original / display text** | Showing results, Sentence-BERT input | Unicode NFKC, typographic quotes and dashes normalised, control characters removed, whitespace collapsed, paragraphs preserved. For SBERT, URLs and e-mails are also removed. |
| **Preprocessed text** | TF-IDF | URL/e-mail removal → lower-casing → NLTK `word_tokenize` → punctuation and special characters removed, hyphenated words split → possessives removed → pure numbers dropped (alphanumerics such as `5g` kept) → stop-words removed → WordNet lemmatisation |

Design decisions:

* **Negations are kept.** `not`, `no`, `nor`, `never`, `against` and `without` are removed from the stop-word list, because dropping them changes meaning.
* **Context-free lemmatisation.** A POS-tag-based lemmatiser was tried first, but short queries give the tagger too little context. "learning" was tagged as a verb in the query *"what is supervised learning"* (lemma `learn`) and as a noun in documents (lemma `learning`), so the terms no longer matched. Every word now goes through a deterministic noun → verb → adjective lemma fallback, so the same word always yields the same lemma.
* **No stemming.** Lemmas stay readable (e.g. `hackers → hacker`, `computers → computer`) and can be shown in the UI.
* Transformer input is **not** stop-word-filtered or lemmatised, because Sentence-BERT relies on full sentence context.

### Chunking (`src/chunking.py`)

* Passages are built from whole sentences (NLTK Punkt) and hold at most **150 words**, well below the model's 256-word-piece limit.
* Consecutive passages overlap by **1 sentence**.
* Section headings start a new passage and are prefixed to it (e.g. *"Spear phishing: …"*).
* Over-long sentences are split on word boundaries. A trailing passage under 20 words is merged into its predecessor.

---

## 6. TF-IDF implementation (`src/tfidf_search.py`)

* `TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=1, max_df=0.95)`, fitted on the preprocessed **title + full text** of each document.
* `max_df=0.95` drops terms that occur in more than 95% of documents (corpus-specific stop-words such as *make*).
* The same vectorizer transforms every passage, which is used to choose the best passage inside each retrieved document.
* Rows are L2-normalised, so the dot product equals cosine similarity.
* The vectorizer and both sparse matrices are saved with **joblib** in `models/tfidf/` and reloaded at start-up.

## 7. Sentence-BERT implementation (`src/semantic_search.py`)

* Model: **`sentence-transformers/all-MiniLM-L6-v2`** (pretrained, 6 layers, 384-dimensional embeddings, about 90 MB), run on **CPU**.
* It is downloaded once into `models/pretrained/`; afterwards it loads offline (`HF_HUB_OFFLINE` is set automatically when the cache exists).
* Each passage is encoded as `"<document title>. <passage>"` so it keeps its document's topic context. Embeddings are L2-normalised.
* Document score = **maximum** passage similarity, and the top passages are shown as evidence.
* Embeddings are stored in `models/embeddings/chunk_embeddings.npy` together with an aligned id list. If the ids don't match, the index refuses to load the embeddings and asks for a rebuild.

## 8. Indexing (`src/indexing.py`)

| Operation | What is (re)computed |
|---|---|
| **Add document(s)** | Only the new documents are extracted, validated, preprocessed, chunked and embedded. Existing embeddings are reused unchanged (verified by a unit test). TF-IDF is re-fitted from cached preprocessed text, because IDF is a corpus-wide statistic; this takes a few seconds. |
| **Remove document(s)** | Their passages and embedding rows are dropped, TF-IDF is re-fitted, and uploaded files are deleted from `data/uploads/`. |
| **Rebuild index** | Every stored document is re-preprocessed, re-chunked and re-embedded. Use it after changing settings. |
| **Reload base dataset** | The Wikipedia snapshot is reloaded (restoring removed articles); uploads are kept. |

* The index stores a **configuration fingerprint** (model name and chunking parameters). If `config/settings.py` changes, the app asks for a rebuild instead of mixing incompatible embeddings.
* All writes are atomic (temporary file + rename).
* Measured build time for the full collection: **about 5.5 minutes** on a 12-thread CPU under WSL2. Nearly all of it is Sentence-BERT encoding of 6,375 passages.

---

## 9. User interface

| Page | Contents |
|---|---|
| **Home** | Indexed documents, passages and uploads; vocabulary size; last update time; descriptions of the search methods; category, length, origin and passage charts; a build button on first run |
| **Search Engine** | Natural-language input, example queries, method choice (TF-IDF / Semantic / Compare both), number of results, category filter. Each result shows title, category, score bar, highlighted best passage, file name and source link, plus extra matching passages and a link to the full document. |
| **Model Comparison** | Both rankings, common documents and documents found by only one method, Jaccard overlap, Spearman rank correlation, a slope chart of rank changes, the score distribution over the whole collection, a per-document table with query-term overlap, and an automatically generated, data-based explanation of the differences |
| **Document Upload** | Multi-file upload with per-file status and progress, a category selector, a results log, and the list of uploaded documents with removal |
| **Document Explorer** | Filterable document table; preview of original text, passages, top TF-IDF terms and preprocessed text; document removal; *Rebuild index* and *Reload base dataset* |
| **Evaluation & Error Analysis** | Model comparison table, results by query type, P@K / R@K / nDCG@K curves, MRR bars, per-query charts, Wilcoxon test, error-analysis cases, the evaluation dataset, and CSV/PNG export |

Light and dark themes are available from Streamlit's *Settings* menu (☰ → Settings → Theme). The custom styling works in both.

---

## 10. Evaluation methodology (`src/evaluation.py`)

### Test set

`data/evaluation/evaluation_queries.json` contains **52 queries with 118 relevance judgements**. They are **manually curated**:

* Labels were written by hand from the content of each article, **before either model was run**, so they do not depend on model output.
* Graded scale: **2** = highly relevant (the article's main topic answers the query); **1** = partially relevant. Unlisted documents count as non-relevant.
* Query types: 32 *paraphrase* (describes a topic without its usual name), 13 *keyword* (technical terms), 6 *question* and 1 *ambiguous* ("virus").
* This is a small project-specific set, not an external benchmark. It can be reviewed and extended in the JSON file.

### Metrics

* **Precision@K** = relevant documents in the top K ÷ K
* **Recall@K** = relevant documents in the top K ÷ all relevant documents
* **MRR** = mean of 1 / rank of the first relevant document, with a cut-off of 10
* **nDCG@K** = DCG@K ÷ ideal DCG@K, with gain 2^grade − 1 and a log₂(rank + 1) discount, so graded labels are used
* K ∈ {1, 3, 5, 10}
* A **Wilcoxon signed-rank test** on per-query nDCG@10 checks whether the difference between methods is statistically significant.

Run it with `python -m src.evaluation` (or open the Evaluation page). Outputs are written to `reports/evaluation_results/`.

---

## 11. Actual evaluation results

Run on 30 Oct 2026 on the 89-document Wikipedia index; all 52 queries were evaluated.

### Model comparison

| Method | P@1 | P@3 | P@5 | P@10 | R@3 | R@5 | R@10 | nDCG@3 | nDCG@5 | nDCG@10 | MRR | Mean latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF | **0.923** | 0.532 | 0.365 | 0.210 | 0.788 | 0.877 | 0.956 | **0.874** | **0.894** | **0.911** | **0.954** | **9 ms** |
| Sentence-BERT | 0.885 | **0.551** | **0.377** | **0.215** | **0.835** | **0.912** | **0.983** | 0.861 | 0.883 | 0.903 | 0.942 | 140 ms |

**Per query (nDCG@10):** Sentence-BERT was better on 18 queries, TF-IDF on 15, and 19 were ties. The mean difference is −0.008 and the Wilcoxon p-value is **0.93**, so the difference is **not statistically significant**.

### By query type

| Query type | n | Method | P@1 | R@10 | nDCG@10 | MRR |
|---|---|---|---|---|---|---|
| keyword | 13 | TF-IDF | **1.000** | 0.974 | **0.958** | **1.000** |
| keyword | 13 | Sentence-BERT | 0.846 | **1.000** | 0.938 | 0.923 |
| paraphrase | 32 | TF-IDF | **0.938** | 0.949 | **0.911** | **0.956** |
| paraphrase | 32 | Sentence-BERT | 0.875 | **0.979** | 0.898 | 0.938 |
| question | 6 | TF-IDF | 0.667 | 0.944 | 0.808 | 0.833 |
| question | 6 | Sentence-BERT | **1.000** | **0.967** | **0.886** | **1.000** |
| ambiguous | 1 | TF-IDF | 1.000 | 1.000 | **0.936** | 1.000 |
| ambiguous | 1 | Sentence-BERT | 1.000 | 1.000 | 0.745 | 1.000 |

### Interpretation

* **Neither method is clearly better on this collection.** TF-IDF is slightly ahead on top-of-ranking metrics (P@1, MRR, nDCG). Sentence-BERT is consistently ahead on **recall** (R@3, R@5, R@10), which means it brings more of the relevant documents into the top results.
* **Sentence-BERT wins on natural questions** (P@1 1.00 vs 0.67). For example, *"how do plants make food from sunlight"* puts *Photosynthesis* first, while TF-IDF ranks *Solar power* first because it shares "sunlight". In *"can a computer think like a human"*, TF-IDF ranks *Expert system* above *Turing test*.
* **TF-IDF wins on keyword queries**, where exact terms such as *TCP/IP* and *lithium-ion* are highly discriminative.
* **Why TF-IDF stays strong on paraphrases here:** the documents are long, comprehensive Wikipedia articles. Even a paraphrased query usually shares some distinctive word with the right article (e.g. *"medicines that kill bacteria"* shares *bacteria* with *Antibiotic*). Document-level TF-IDF aggregates evidence over 2,000 to 15,000 words.
* **Sentence-BERT's main weakness here is max-pooling.** One passage can be close to the query even when the document is about something else. For example, *"software that locks your files and demands payment"* ranks *Internet* first, and *"flooding a website with traffic"* ranks *Large language model* first.
* **Latency:** TF-IDF takes about 9 ms per query versus about 140 ms for Sentence-BERT (query encoding plus 6,375 dot products on CPU).

Static charts are in `reports/evaluation_results/` (`metrics_at_k.png`, `mrr_comparison.png`, `per_query_ndcg.png`); the tables are in CSV and JSON.

## 12. Error analysis (`src/error_analysis.py`)

The Evaluation page selects cases automatically from the evaluation run:

* queries whose **top result is irrelevant** (10 query–method cases in the run above);
* **highly relevant documents missing from the top 5** (6 cases; e.g. *Firewall (computing)* for the hacker query, ranked 43rd by TF-IDF and 21st by SBERT);
* queries where **TF-IDF performs better** (10) and where **Sentence-BERT performs better** (13), each with both top-3 lists.

For each case, data-driven **heuristic** signals suggest a likely cause: a short or ambiguous query, missing vocabulary (terms absent from the collection), uninformative terms (filtered by `max_df`), low overlap between query terms and the relevant documents (vocabulary mismatch), a short or very long relevant document, a topical neighbour in the same category, cross-topic confusion, keyword overlap without relevance, or semantic drift from a single passage. These are indicators, not proofs.

---

## 13. Limitations

* **Small, self-labelled test set.** 52 queries labelled by one annotator. Groups such as *question* (6) and *ambiguous* (1) are too small for firm conclusions, and the significance test shows the overall difference is not significant.
* **Collection size.** 89 long articles, each about a distinct topic, which favours document-level TF-IDF. Results may differ on short or more homogeneous documents.
* **No OCR.** Scanned or image-only PDFs are rejected with an explanatory message.
* **English only.** NLTK resources and `all-MiniLM-L6-v2` target English.
* **Lemmatiser quirks.** Context-free WordNet lemmatisation produces some odd lemmas (*physics → physic*, *species → specie*). They are applied the same way to queries and documents, so matching still works.
* **Model limits.** The 256-word-piece input limit is handled by chunking. The model is not fine-tuned on this domain.
* **TF-IDF refit on every change.** This is fast for this collection size; at larger scale an incremental or approximate scheme would be needed.
* **Brute-force similarity.** Dot products over all passages are fine for thousands of passages; millions would need a vector index (e.g. FAISS).
* **Single-user design.** The index is shared within one Streamlit server process; there is no authentication or multi-user isolation.
* **Python 3.10.** Dependency resolution for 3.10 was checked, but the project was executed and tested on Python 3.11.

---

## 14. Installation and execution

### 14.1 Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### 14.2 Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

*Note:* on some Debian/Ubuntu systems `python3 -m venv` fails until `python3-venv` is installed. Alternatively, with [uv](https://docs.astral.sh/uv/): `uv venv --python 3.11 --seed .venv`.

### 14.3 First run

1. Open the URL printed by Streamlit (default `http://localhost:8501`).
2. On **Home**, click **Build search index**. The index is built from the Wikipedia snapshot in `data/raw/wikipedia/`; if the snapshot is missing, it is downloaded first. The NLTK resources (`nltk_data/`) and the model (`models/pretrained/`) download automatically on first use. Building takes a few minutes on CPU.
3. Later starts load the saved index in seconds.

Equivalent command-line steps (with the virtual environment activated):

```bash
python -m src.data_loader --download     # (re)download the Wikipedia snapshot
python -m src.data_loader --stats        # dataset statistics
python -m src.indexing --build           # build the index
python -m src.indexing --search "How can I protect my computer from hackers?" --method semantic
python -m src.evaluation                 # evaluate both methods, write reports/
python -m pytest                         # run all tests
```

### 14.4 VS Code

`.vscode/settings.json` points to `.venv/bin/python` (Linux/WSL). On native Windows, choose **Python: Select Interpreter → `.venv\Scripts\python.exe`**; the Python extension detects the workspace `.venv` automatically.

### 14.5 Requirements files

* `requirements.txt` uses compatible version ranges that resolve on Python 3.10 and 3.11. It adds the PyTorch CPU wheel index, so Linux does not download the multi-GB CUDA build.
* `requirements-lock.txt` pins the exact versions that were tested (Python 3.11.17).

---

## 15. Project structure

```
├── app.py                     # Streamlit entry point, navigation, Home dashboard
├── config/settings.py         # paths, model name, chunking & TF-IDF parameters
├── src/
│   ├── data_loader.py         # Wikipedia download, cleaning, dedupe, sample fallback
│   ├── document_processor.py  # PDF/TXT/DOCX detection, extraction, validation
│   ├── preprocessing.py       # display/SBERT cleaning + classical TF-IDF pipeline
│   ├── chunking.py            # sentence-based passages
│   ├── tfidf_search.py        # TF-IDF fit/save/load, cosine scoring
│   ├── semantic_search.py     # Sentence-BERT loading, encoding, max-pooling
│   ├── indexing.py            # SearchIndex: persistence, add/remove/rebuild, search
│   ├── comparison.py          # TF-IDF vs SBERT comparison + explanations
│   ├── evaluation.py          # P@K, R@K, MRR, nDCG@K, reports
│   ├── error_analysis.py      # failure cases + heuristic diagnostics
│   ├── visualizations.py      # Plotly figures
│   └── ui.py                  # shared Streamlit helpers (cache, highlighting, cards)
├── pages/                     # Upload, Search, Comparison, Evaluation, Explorer
├── data/
│   ├── raw/wikipedia/         # dataset snapshot + manifest (sources, revisions)
│   ├── raw/sample/            # labelled fallback sample documents
│   ├── evaluation/            # manually curated evaluation queries
│   ├── uploads/               # user-uploaded files
│   └── processed/             # documents.json, chunks.json, index_info.json
├── models/
│   ├── tfidf/                 # vectorizer + matrices (joblib)
│   ├── embeddings/            # passage embeddings (.npy) + ids
│   └── pretrained/            # cached Hugging Face model
├── reports/evaluation_results # CSV / JSON / PNG evaluation outputs
├── tests/                     # pytest suite (70 tests)
├── requirements.txt / requirements-lock.txt
└── .streamlit/config.toml, .vscode/settings.json, pytest.ini
```

## 16. Testing

`python -m pytest` runs **70 tests**:

* `test_preprocessing.py`: tokenisation, URL and special-character handling, stop-words with negations kept, lemmatisation consistency, empty input, chunk size, overlap, coverage and headings.
* `test_document_processing.py`: real generated PDF (multi-page, metadata), DOCX (paragraphs, tables, title) and TXT (UTF-8, UTF-8 BOM, UTF-16, cp1252); rejection of empty, too short, non-text, fake-extension and corrupt files; dataset de-duplication; Wikipedia cleaning.
* `test_search.py` (real Sentence-BERT model, isolated temp storage): TF-IDF ranking and out-of-vocabulary handling, paraphrase retrieval, embedding normalisation, category filter, persistence round-trip, incremental add that reuses embeddings, duplicate detection, removal and alignment, rebuild reproducibility, upload of all three formats plus deletion, the fingerprint check, the comparison module, metric formulas, evaluation and error analysis.
* `test_app.py` (Streamlit `AppTest`, read-only against the built index): Home metrics, every page renders, all three search modes, comparison, explorer deep link. Skipped automatically if the index has not been built.
