"""Upload PDF / TXT / DOCX documents and add them to the search index."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import settings
from src.document_processor import DocumentValidationError
from src.ui import format_timestamp, get_index

index = get_index()
st.title("📤 Document Upload")
st.markdown(
    "Upload your own **PDF**, **TXT** or **DOCX** files. Each file is validated, its text extracted, split "
    "into passages, embedded with Sentence-BERT and added to the TF-IDF index. Existing documents are not "
    "re-processed; only the TF-IDF weights are re-fitted because IDF depends on the whole collection."
)
if index.needs_rebuild:
    st.warning("The index needs a rebuild before new documents can be added. Open **Home** in the sidebar.")
    st.stop()

st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("upload_log", [])

with st.container(border=True):
    files = st.file_uploader(
        "Choose files", type=[e.lstrip(".") for e in settings.SUPPORTED_EXTENSIONS],
        accept_multiple_files=True, key=f"uploader_{st.session_state['uploader_key']}",
        help=f"Maximum {settings.MAX_UPLOAD_MB} MB per file. Scanned (image-only) PDFs are not supported.",
    )
    existing = sorted({d["category"] for d in index.documents} | set(settings.CATEGORIES))
    options = [settings.UPLOAD_CATEGORY_DEFAULT] + [c for c in existing if c != settings.UPLOAD_CATEGORY_DEFAULT] + ["Other…"]
    c1, c2 = st.columns(2)
    choice = c1.selectbox("Category for these documents", options)
    custom = c2.text_input("Custom category", disabled=choice != "Other…", placeholder="e.g. Finance")
    category = (custom.strip() or settings.UPLOAD_CATEGORY_DEFAULT) if choice == "Other…" else choice
    start = st.button("Upload and index", type="primary", icon=":material/cloud_upload:", disabled=not files)

if start and files:
    log = []
    overall = st.progress(0.0, text="Starting ...")
    for i, uploaded in enumerate(files):
        with st.status(f"Processing **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)", expanded=True) as status:
            step = st.empty()

            def on_progress(fraction: float, message: str, _i=i) -> None:
                step.caption(message)
                overall.progress((_i + fraction) / len(files), text=f"File {_i + 1}/{len(files)}: {message}")

            try:
                report = index.add_uploaded_file(uploaded.name, uploaded.getvalue(), category=category,
                                                 progress=on_progress)
            except DocumentValidationError as exc:
                status.update(label=f"❌ {uploaded.name}: rejected", state="error")
                st.error(str(exc))
                log.append({"file": uploaded.name, "status": "rejected", "details": str(exc)})
                continue
            except Exception as exc:  # unexpected parser failure: report, keep going
                status.update(label=f"❌ {uploaded.name}: failed", state="error")
                st.error(f"Unexpected error while processing the file: {exc}")
                log.append({"file": uploaded.name, "status": "error", "details": str(exc)})
                continue
            if report.added:
                doc = index.get_document(report.added[0])
                st.write(f"Title: **{doc['title']}** · {doc['word_count']:,} words · "
                         f"{doc['chunk_count']} passages · category *{doc['category']}*")
                status.update(label=f"✅ {uploaded.name}: indexed", state="complete", expanded=False)
                log.append({"file": uploaded.name, "status": "indexed",
                            "details": f"{doc['word_count']:,} words, {doc['chunk_count']} passage(s)"})
            else:
                reason = "; ".join(report.skipped) or "not added"
                st.info(reason)
                status.update(label=f"⚠️ {uploaded.name}: skipped", state="complete", expanded=False)
                log.append({"file": uploaded.name, "status": "skipped", "details": reason})
    overall.progress(1.0, text="Done")
    st.session_state["upload_log"] = log
    st.session_state.pop("evaluation", None)
    st.session_state["uploader_key"] += 1  # clears the uploader so files are not indexed twice
    st.rerun()

if st.session_state["upload_log"]:
    st.subheader("Last upload")
    log_df = pd.DataFrame(st.session_state["upload_log"])
    indexed = (log_df["status"] == "indexed").sum()
    (st.success if indexed else st.warning)(f"{indexed} of {len(log_df)} file(s) indexed.")
    st.dataframe(log_df, hide_index=True, width="stretch")

st.subheader("Uploaded documents in the index")
uploads = [d for d in index.documents if d.get("origin") == "upload"]
if not uploads:
    st.caption("No uploaded documents yet.")
else:
    table = pd.DataFrame([{
        "doc_id": d["doc_id"], "title": d["title"], "category": d["category"], "file": d["file_name"],
        "words": d["word_count"], "passages": d["chunk_count"], "added": format_timestamp(d["added_at"]),
    } for d in uploads])
    selection = st.dataframe(table, hide_index=True, width="stretch", on_select="rerun",
                             selection_mode="multi-row", column_config={"doc_id": None})
    rows = selection.selection.rows
    if st.button(f"Remove {len(rows)} selected document(s)", icon=":material/delete:", disabled=not rows):
        removed = index.remove_documents([table.iloc[r]["doc_id"] for r in rows])
        st.session_state.pop("evaluation", None)
        st.toast(f"Removed {len(removed)} document(s) and updated the index.")
        st.rerun()
    st.caption("Select rows to remove them. Stored files in `data/uploads/` are deleted together with "
               "their passages and embeddings.")
