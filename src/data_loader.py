"""Dataset collection and loading.

Primary dataset
---------------
Plain-text versions of public English Wikipedia articles, downloaded through
the official MediaWiki Action API (``prop=extracts&explaintext=1``). Wikipedia
text is licensed under CC BY-SA 4.0; every document keeps its source URL and
revision id so the exact text can be traced back.

Fallback dataset
----------------
If the Wikipedia snapshot is not present and cannot be downloaded (e.g. no
internet access), a small set of sample documents written specifically for
this project is used instead. Those documents are clearly labelled as
``origin="sample"`` and are *not* presented as external documents.

Usage::

    python -m src.data_loader --download      # fetch / refresh the snapshot
    python -m src.data_loader --stats         # print dataset statistics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import settings

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "SemanticDocumentSearch/1.0 (MCA AI&ML capstone project; educational use) "
    "Python-urllib"
)
WIKI_LICENSE = "CC BY-SA 4.0"
WIKI_SOURCE = "Wikipedia"

# Articles to download, grouped by category. Titles are canonical English
# Wikipedia page names; redirects are resolved by the API.
WIKIPEDIA_ARTICLES: dict[str, list[str]] = {
    "Artificial Intelligence": [
        "Artificial intelligence", "Natural language processing", "Computer vision",
        "Expert system", "Large language model", "Chatbot", "Turing test",
        "Artificial general intelligence", "Ethics of artificial intelligence",
        "Speech recognition", "Machine translation",
    ],
    "Machine Learning": [
        "Machine learning", "Deep learning", "Neural network (machine learning)",
        "Support vector machine", "Random forest", "Reinforcement learning",
        "Supervised learning", "Unsupervised learning", "Decision tree learning",
        "Gradient descent", "Overfitting", "Transformer (deep learning architecture)",
    ],
    "Cybersecurity": [
        "Computer security", "Malware", "Firewall (computing)", "Phishing",
        "Ransomware", "Encryption", "Antivirus software", "Computer virus",
        "Denial-of-service attack", "Multi-factor authentication",
        "Intrusion detection system", "Social engineering (security)",
        "Password strength",
    ],
    "Healthcare": [
        "Health care", "Diabetes", "Vaccine", "Cancer", "Hypertension",
        "Telehealth", "Public health", "Mental health", "Antibiotic",
        "Influenza", "Obesity",
    ],
    "Education": [
        "Education", "Educational technology", "Distance education", "Curriculum",
        "Higher education", "Literacy", "Homeschooling",
        "Massive open online course", "Standardized test",
        "Early childhood education",
    ],
    "Technology": [
        "Internet", "Smartphone", "Cloud computing", "Blockchain",
        "Internet of things", "5G", "Semiconductor", "Quantum computing",
        "Virtual reality", "Database", "Electric vehicle",
    ],
    "Environment": [
        "Climate change", "Renewable energy", "Deforestation", "Air pollution",
        "Biodiversity", "Recycling", "Plastic pollution", "Water pollution",
        "Solar power", "Ocean acidification", "Wind power",
    ],
    "Science": [
        "Photosynthesis", "Black hole", "DNA", "General relativity",
        "Periodic table", "Evolution", "Quantum mechanics", "Plate tectonics",
        "Big Bang", "Cell (biology)",
    ],
}

# Trailing Wikipedia sections that contain only links / citations.
_TRAILING_SECTIONS = {
    "see also", "references", "notes", "external links", "further reading",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "works cited", "general and cited references", "cited sources",
}
_HEADING_RE = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "document"


def content_hash(text: str) -> str:
    """Hash of whitespace/case-normalised text, used for duplicate detection."""
    normalised = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _strip_braced(text: str, marker: str) -> str:
    """Remove ``{marker ...}`` blocks with balanced braces (LaTeX residue)."""
    out, i = [], 0
    while (j := text.find(marker, i)) != -1:
        out.append(text[i:j])
        depth, k = 0, j  # text[j] is the opening brace of the block
        while k < len(text):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                break
            k += 1
        i = k + 1
    out.append(text[i:])
    return "".join(out)


def _trailing_cut_line(lines: list[str]) -> int:
    """Index of the line where trailing link/citation sections start.

    Only the *final contiguous run* of level-2 sections whose headings are all
    link/citation sections is removed. This avoids cutting real content
    sections that happen to share a name, e.g. "Sources" of air pollution.
    """
    level2 = [
        (i, m.group(2).strip().lower())
        for i, line in enumerate(lines)
        if (m := _HEADING_RE.match(line.strip())) and len(m.group(1)) == 2
    ]
    cut = len(lines)
    for i, heading in reversed(level2):
        if heading not in _TRAILING_SECTIONS:
            break
        cut = i
    return cut


def clean_wikipedia_text(text: str) -> str:
    """Convert a MediaWiki plain-text extract into clean prose.

    * drops the trailing link/citation sections (See also, References, ...)
    * turns ``== Heading ==`` markers into plain heading lines
    * removes LaTeX ``{\\displaystyle ...}`` residue left by math markup
    """
    for marker in ("{\\displaystyle", "{\\textstyle"):
        text = _strip_braced(text, marker)

    lines = text.splitlines()
    lines_out: list[str] = []
    for line in lines[: _trailing_cut_line(lines)]:
        match = _HEADING_RE.match(line.strip())
        if match:
            lines_out.extend(["", match.group(2).strip(), ""])
        else:
            lines_out.append(line.rstrip())

    cleaned = "\n".join(lines_out)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _wiki_request(title: str, timeout: int = 30, retries: int = 5) -> dict:
    """Query the MediaWiki API, backing off politely on HTTP 429 / 5xx."""
    params = {
        "action": "query", "format": "json", "formatversion": "2",
        "prop": "extracts|info", "inprop": "url", "explaintext": "1",
        "exsectionformat": "wiki", "redirects": "1", "titles": title,
        "maxlag": "5",
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    wait = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after and retry_after.isdigit() else wait
            time.sleep(min(sleep_for, 120))
            wait *= 2
    raise RuntimeError("unreachable")


def download_wikipedia_dataset(delay: float = 2.0, verbose: bool = True, force: bool = False) -> dict:
    """Download all configured articles into ``data/raw/wikipedia``.

    Returns a summary dict with counts of downloaded, failed, duplicate and
    empty articles. Writes ``manifest.json`` describing every stored file.
    """
    manifest_path = settings.WIKI_DIR / "manifest.json"
    # Resume support: keep articles that were already stored by a previous run.
    manifest: list[dict] = []
    if manifest_path.exists() and not force:
        manifest = [
            entry for entry in json.loads(manifest_path.read_text(encoding="utf-8"))
            if (settings.WIKI_DIR / entry["relative_path"]).exists()
        ]
    done_titles = {entry["requested_title"] for entry in manifest if "requested_title" in entry}
    seen_page_ids: set[int] = {entry["page_id"] for entry in manifest}
    seen_hashes: set[str] = {
        content_hash((settings.WIKI_DIR / entry["relative_path"]).read_text(encoding="utf-8"))
        for entry in manifest
    }
    summary = {"requested": 0, "stored": 0, "already_present": 0,
               "failed": [], "duplicates": [], "empty": []}

    for category, titles in WIKIPEDIA_ARTICLES.items():
        category_dir = settings.WIKI_DIR / slugify(category)
        category_dir.mkdir(parents=True, exist_ok=True)
        for title in titles:
            summary["requested"] += 1
            if title in done_titles:
                summary["already_present"] += 1
                continue
            time.sleep(delay)
            try:
                payload = _wiki_request(title)
                page = payload["query"]["pages"][0]
            except Exception as exc:  # network / API errors
                summary["failed"].append(f"{title}: {exc}")
                if verbose:
                    print(f"  [FAILED] {title}: {exc}")
                continue
            if page.get("missing") or "extract" not in page:
                summary["failed"].append(f"{title}: page missing")
                if verbose:
                    print(f"  [MISSING] {title}")
                continue

            page_id = page["pageid"]
            raw_dir = settings.WIKI_DIR / "raw_extracts"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"{slugify(page['title'])}.txt").write_text(page["extract"], encoding="utf-8")
            text = clean_wikipedia_text(page["extract"])
            digest = content_hash(text)
            if page_id in seen_page_ids or digest in seen_hashes:
                summary["duplicates"].append(title)
                if verbose:
                    print(f"  [DUPLICATE] {title} -> {page['title']}")
                continue
            if len(text) < settings.MIN_DOCUMENT_CHARS:
                summary["empty"].append(title)
                if verbose:
                    print(f"  [EMPTY] {title}")
                continue
            seen_page_ids.add(page_id)
            seen_hashes.add(digest)

            file_name = f"{slugify(page['title'])}.txt"
            (category_dir / file_name).write_text(text, encoding="utf-8")
            manifest.append({
                "doc_id": f"wiki_{slugify(page['title'])}",
                "requested_title": title,
                "title": page["title"],
                "category": category,
                "source": f"{WIKI_SOURCE} ({WIKI_LICENSE})",
                "source_url": page.get("fullurl", ""),
                "page_id": page_id,
                "revision_id": page.get("lastrevid"),
                "file_name": file_name,
                "relative_path": f"{slugify(category)}/{file_name}",
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            summary["stored"] += 1
            if verbose:
                print(f"  [OK] {category:<24} {page['title']} ({len(text):,} chars)")
            # Persist after every article so an interrupted run can resume.
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["manifest"] = str(manifest_path)
    return summary


def _load_from_manifest(base_dir: Path, origin: str) -> list[dict]:
    manifest_path = base_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents = []
    for entry in entries:
        path = base_dir / entry["relative_path"]
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        documents.append({**entry, "content": content, "origin": origin})
    return documents


def load_wikipedia_documents() -> list[dict]:
    return _load_from_manifest(settings.WIKI_DIR, origin="dataset")


def load_sample_documents() -> list[dict]:
    return _load_from_manifest(settings.SAMPLE_DIR, origin="sample")


def clean_document_records(documents: list[dict]) -> tuple[list[dict], dict]:
    """Validate records: drop missing/empty content and exact duplicates."""
    report = {"input": len(documents), "empty": [], "duplicates": [], "kept": 0}
    cleaned, seen_ids, seen_hashes = [], set(), set()
    for doc in documents:
        text = (doc.get("content") or "").strip()
        if len(text) < settings.MIN_DOCUMENT_CHARS:
            report["empty"].append(doc.get("doc_id", "?"))
            continue
        digest = content_hash(text)
        if digest in seen_hashes or doc["doc_id"] in seen_ids:
            report["duplicates"].append(doc["doc_id"])
            continue
        seen_hashes.add(digest)
        seen_ids.add(doc["doc_id"])
        doc = {**doc, "content": text, "content_hash": digest}
        doc.setdefault("title", doc["doc_id"])
        doc.setdefault("category", "Uncategorised")
        doc.setdefault("source", "Unknown")
        doc.setdefault("file_name", "")
        cleaned.append(doc)
    report["kept"] = len(cleaned)
    return cleaned, report


def load_base_dataset(allow_download: bool = True) -> tuple[list[dict], str]:
    """Return the base document collection and a label describing its origin.

    Order of preference: existing Wikipedia snapshot -> fresh download ->
    clearly labelled sample documents.
    """
    documents = load_wikipedia_documents()
    if not documents and allow_download:
        try:
            download_wikipedia_dataset(verbose=False)
            documents = load_wikipedia_documents()
        except Exception:
            documents = []
    if documents:
        cleaned, _ = clean_document_records(documents)
        return cleaned, "wikipedia"
    cleaned, _ = clean_document_records(load_sample_documents())
    return cleaned, "sample"


def dataset_statistics(documents: list[dict]) -> dict:
    words = [len(d["content"].split()) for d in documents]
    by_category: dict[str, int] = {}
    for doc in documents:
        by_category[doc["category"]] = by_category.get(doc["category"], 0) + 1
    return {
        "documents": len(documents),
        "categories": by_category,
        "total_words": sum(words),
        "min_words": min(words) if words else 0,
        "max_words": max(words) if words else 0,
        "mean_words": round(sum(words) / len(words), 1) if words else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true", help="download the Wikipedia snapshot")
    parser.add_argument("--force", action="store_true", help="re-download articles already present")
    parser.add_argument("--stats", action="store_true", help="print dataset statistics")
    args = parser.parse_args(argv)

    if args.download:
        print(f"Downloading {sum(map(len, WIKIPEDIA_ARTICLES.values()))} Wikipedia articles ...")
        summary = download_wikipedia_dataset(force=args.force)
        print(json.dumps({k: v for k, v in summary.items()}, indent=2))
    if args.stats or not args.download:
        documents, origin = load_base_dataset(allow_download=False)
        print(f"Dataset origin: {origin}")
        print(json.dumps(dataset_statistics(documents), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
