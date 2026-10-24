import io
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List

import pdfplumber
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Read the key from the .env file that sits next to this script.
# The key is not written in this file on purpose.
load_dotenv()

# These two names are the Gemini models this page calls.
# The first one writes the answer. The second one turns text into numbers.
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# How the page cuts a document up.
# A "child" is the small piece we search.
# A "parent" is the bigger piece around it that we send to the model.
CHILD_WORDS = 120
OVERLAP_WORDS = 30
PARENT_WORDS = 320

# Stop after this many chunks so a huge file does not spend the whole API quota.
MAX_CHUNKS_PER_FILE = 80

# How many chunks we mix together, and how many we finally send to Gemini.
FUSE_LIMIT = 20
TOP_K = 5

# Rough size limit for the text we paste into the prompt.
# About 4 characters is one token, so 6000 characters is near 1500 tokens.
CONTEXT_CHAR_BUDGET = 6000


# ── Talk to Gemini ───────────────────────────────────────────────────────────
def get_client() -> genai.Client:
    """Make a Gemini client from the key in .env."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Put it in the .env file.")
    return genai.Client(api_key=api_key)


def embed_texts(texts: List[str], task_type: str) -> List[List[float]]:
    """
    Turn each string into a vector (a long list of numbers).

    task_type tells Gemini what the text is for:
    - RETRIEVAL_DOCUMENT for a chunk from a file
    - RETRIEVAL_QUERY for the user's question

    Same model, but the two task types put questions and paragraphs
    into a space we can compare.
    """
    client = get_client()
    vectors: List[List[float]] = []

    # Send a few strings at a time. One giant request is easier to get rejected.
    batch_size = 8
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=768,
            ),
        )
        for item in response.embeddings:
            vectors.append([float(value) for value in item.values])

    return vectors


# ── Read the uploaded file ───────────────────────────────────────────────────
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Pull the selectable text out of a PDF. A scanned photo-PDF often has none."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Pull paragraph text out of a Word file. Tables are left out in this version."""
    doc = Document(io.BytesIO(file_bytes))
    lines = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    return "\n".join(lines)


def read_upload(uploaded_file) -> str:
    """
    Look at the file name and pick a reader.
    Supports PDF, DOCX, and TXT, same as a simple student uploader.
    """
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    elif name.endswith(".docx"):
        text = extract_text_from_docx(file_bytes)
    elif name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("No text found. If this is a scanned PDF, it needs real text, not a photo of a page.")
    return text


# ── Cut the document into parent and child chunks ────────────────────────────
def split_into_chunks(text: str, source_name: str) -> List[dict]:
    """
    Cut the document into overlapping pieces.

    We search the small child chunk, because a short piece matches a question more cleanly.
    We later show Gemini the parent, which is the child plus the words around it,
    so the answer has a bit more context than the exact match.

    The id is the file name plus the chunk number. The same file always gets the
    same ids, which is what a vector database would store.
    """
    words = text.split()
    chunks = []
    start = 0
    index = 0

    while start < len(words):
        end = min(start + CHILD_WORDS, len(words))

        # Center a wider window on this child. That window is the parent.
        extra = PARENT_WORDS - (end - start)
        parent_start = max(0, start - extra // 2)
        parent_end = min(len(words), parent_start + PARENT_WORDS)

        chunks.append(
            {
                "id": f"{source_name}-{index}",
                "source": source_name,
                "text": " ".join(words[start:end]),
                "parent": " ".join(words[parent_start:parent_end]),
            }
        )
        index += 1

        if end == len(words):
            break
        # A very long file stops here so we do not embed hundreds of chunks.
        if index >= MAX_CHUNKS_PER_FILE:
            chunks[-1]["cut_short"] = True
            break

        # Step forward, but keep OVERLAP_WORDS so a sentence on the edge is not cut in half.
        next_start = end - OVERLAP_WORDS
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


# ── Search: meaning (dense) + words (BM25) ───────────────────────────────────
def tokenize(text: str) -> List[str]:
    """Lowercase the text and keep only words and numbers."""
    return re.findall(r"[a-z0-9]+", text.lower())


def cosine(left: List[float], right: List[float]) -> float:
    """
    Cosine similarity. 1 means the two vectors point the same way.
    0 means they are unrelated. We use this to compare a question with a chunk.
    """
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def keyword_scores(query: str, chunks: List[dict]) -> List[float]:
    """
    A small BM25 score.

    This is the "same words" search. It scores a chunk higher when the
    question's words appear in it, and it turns the score down when a word
    is in almost every chunk (those words are not helpful).
    """
    tokenized = [tokenize(chunk["text"]) for chunk in chunks]
    doc_lengths = [len(tokens) or 1 for tokens in tokenized]
    average_length = sum(doc_lengths) / len(doc_lengths)

    doc_frequency: Counter = Counter()
    for tokens in tokenized:
        doc_frequency.update(set(tokens))

    total_docs = len(chunks)
    query_terms = tokenize(query)
    k1 = 1.5
    b = 0.75
    scores = []

    for tokens, length in zip(tokenized, doc_lengths):
        term_frequency = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            # idf: rare words add more than words that show up everywhere
            rare = (total_docs - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5)
            idf = math.log(1 + rare)
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += idf * (frequency * (k1 + 1)) / denominator
        scores.append(score)

    return scores


def fuse_ranks(dense_order: List[int], keyword_order: List[int]) -> Dict[int, float]:
    """
    Reciprocal rank fusion.

    Dense scores and keyword scores are different kinds of numbers, so we
    do not add them. We add 1 / (60 + rank) for each list instead.
    60 is the usual constant. It keeps the first place from always winning alone.
    """
    fused: Dict[int, float] = {}
    for order in (dense_order, keyword_order):
        for rank, chunk_index in enumerate(order):
            fused[chunk_index] = fused.get(chunk_index, 0.0) + 1.0 / (60 + rank + 1)
    return fused


def search_chunks(question: str, chunks: List[dict]) -> List[dict]:
    """
    Hybrid search.

    1. Embed the question and compare it with every stored chunk (meaning).
    2. Score the same chunks with BM25 (words).
    3. Fuse the two rankings and keep the best few.
    """
    if not chunks:
        return []

    query_vector = embed_texts([question], "RETRIEVAL_QUERY")[0]
    cosines = [cosine(query_vector, chunk["embedding"]) for chunk in chunks]
    keywords = keyword_scores(question, chunks)

    dense_order = sorted(range(len(chunks)), key=lambda i: cosines[i], reverse=True)[:FUSE_LIMIT]
    keyword_order = sorted(range(len(chunks)), key=lambda i: keywords[i], reverse=True)
    keyword_order = [i for i in keyword_order if keywords[i] > 0][:FUSE_LIMIT]

    fused = fuse_ranks(dense_order, keyword_order)
    best_indexes = sorted(fused, key=lambda i: fused[i], reverse=True)[:TOP_K]

    hits = []
    for chunk_index in best_indexes:
        hit = dict(chunks[chunk_index])
        hit.pop("embedding", None)
        hit["cosine"] = round(cosines[chunk_index], 3)
        hit["keyword_score"] = round(keywords[chunk_index], 3)
        hit["fusion"] = round(fused[chunk_index], 4)
        hits.append(hit)
    return hits


def rerank_hits(question: str, hits: List[dict]) -> List[dict]:
    """
    Optional second pass.

    A product like Cohere Rerank reads the question and each chunk again and
    reorders them. This project does not have a Cohere key, so the checkbox
    asks Gemini to do a small version of that job. If the reply is not usable,
    we keep the original order.
    """
    if len(hits) < 2:
        return hits

    listing = "\n\n".join(f"ID: {hit['id']}\n{hit['text']}" for hit in hits)
    prompt = (
        "Order these passages by how useful they are for answering the question.\n"
        "Return ONLY a JSON list of the ids, best first. Do not add any other text.\n\n"
        f"Question: {question}\n\n"
        f"Passages:\n{listing}"
    )

    try:
        client = get_client()
        response = client.models.generate_content(model=CHAT_MODEL, contents=prompt)
        raw = (response.text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json", "", 1).strip()
        ordered_ids = json.loads(raw)
        if not isinstance(ordered_ids, list):
            return hits
        by_id = {hit["id"]: hit for hit in hits}
        reranked = [by_id[item_id] for item_id in ordered_ids if item_id in by_id]
        for hit in hits:
            if hit["id"] not in {item["id"] for item in reranked}:
                reranked.append(hit)
        return reranked or hits
    except Exception:
        # A bad reply or a failed call should not throw away the search results.
        return hits


# ── Build the prompt and stream the answer ───────────────────────────────────
def build_context(hits: List[dict]) -> tuple:
    """
    Paste parent chunks into the prompt until the character budget is used up.

    Parents overlap, so we skip a parent that is already inside one we kept.
    The model then sees a few useful sections, not the whole file.
    """
    blocks = []
    used_hits = []
    used_chars = 0
    kept_parents: List[str] = []

    for hit in hits:
        parent = hit["parent"]
        already_covered = any(parent in older or older in parent for older in kept_parents)
        if already_covered:
            continue

        block = f"[{hit['source']} | {hit['id']}]\n{parent}"
        if used_chars + len(block) > CONTEXT_CHAR_BUDGET:
            block = f"[{hit['source']} | {hit['id']}]\n{hit['text']}"
            if used_chars + len(block) > CONTEXT_CHAR_BUDGET:
                break

        blocks.append(block)
        used_hits.append(hit)
        kept_parents.append(parent)
        used_chars += len(block)

    return "\n\n".join(blocks), used_hits


def format_history(messages: List[dict]) -> str:
    """Keep a short memory of the chat so a follow-up question still makes sense."""
    lines = []
    for message in messages[-6:]:
        speaker = "Student" if message["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {message['content'][:400]}")
    if not lines:
        return "(no earlier messages)"
    return "\n".join(lines)


def stream_answer(question: str, context: str, history: List[dict]):
    """
    Ask Gemini to answer from the retrieved passages only, and yield the
    reply a piece at a time so the page can stream it.
    """
    prompt = f"""
You are a document assistant for a student project.
Answer the question using only the source passages below.
If the passages do not contain the answer, say: "I cannot find that in the uploaded documents."
Do not use outside knowledge.
When you use a fact, mention the file name it came from.
Keep the answer short and clear.

Source passages:
{context}

Earlier chat:
{format_history(history)}

Question:
{question}
""".strip()

    client = get_client()
    stream = client.models.generate_content_stream(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2),
    )
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


# ── The page ─────────────────────────────────────────────────────────────────
def init_state() -> None:
    """Streamlit reruns this whole file on every click. Session state is what we keep."""
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "status" not in st.session_state:
        st.session_state.status = "No documents indexed yet."
    if "use_rerank" not in st.session_state:
        st.session_state.use_rerank = False


def index_uploads(uploaded_files) -> None:
    """Read, chunk, embed, and store. Old chunks from the same file name are replaced."""
    new_names = {uploaded.name for uploaded in uploaded_files}
    kept = [chunk for chunk in st.session_state.chunks if chunk["source"] not in new_names]
    fresh_chunks: List[dict] = []

    for uploaded in uploaded_files:
        try:
            text = read_upload(uploaded)
        except Exception as error:
            st.sidebar.error(f"{uploaded.name}: {error}")
            continue

        pieces = split_into_chunks(text, uploaded.name)
        if not pieces:
            st.sidebar.warning(f"{uploaded.name}: nothing to index.")
            continue
        if any(piece.get("cut_short") for piece in pieces):
            st.sidebar.warning(f"{uploaded.name}: only the first {MAX_CHUNKS_PER_FILE} chunks were indexed.")
        for piece in pieces:
            piece.pop("cut_short", None)
        fresh_chunks.extend(pieces)

    if not fresh_chunks:
        return

    progress = st.sidebar.progress(0.0, text="Creating embeddings...")
    vectors: List[List[float]] = []
    batch_size = 8
    texts = [chunk["text"] for chunk in fresh_chunks]

    for start in range(0, len(texts), batch_size):
        vectors.extend(embed_texts(texts[start : start + batch_size], "RETRIEVAL_DOCUMENT"))
        progress.progress(min(1.0, (start + batch_size) / len(texts)))

    for chunk, vector in zip(fresh_chunks, vectors):
        chunk["embedding"] = vector

    st.session_state.chunks = kept + fresh_chunks
    sources = sorted({chunk["source"] for chunk in st.session_state.chunks})
    st.session_state.status = f"Indexed {len(st.session_state.chunks)} chunks from {', '.join(sources)}."
    progress.empty()


def draw_sidebar() -> None:
    st.sidebar.title("Your documents")
    st.sidebar.caption("Upload a file, then click Index. After that, ask a question in the chat.")
    st.sidebar.markdown(
        """
**After you ask, this page:**
1. Turns the question into numbers and words
2. Scores every chunk both ways
3. Mixes the two lists
4. Asks Gemini to answer from the best chunks only
"""
    )

    uploads = st.sidebar.file_uploader(
        "Upload PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Try sample_notes.txt if you want a file that already has answers in it.",
    )

    if st.sidebar.button("Index documents", use_container_width=True):
        if not uploads:
            st.sidebar.error("Upload a file first.")
        elif not os.getenv("GEMINI_API_KEY"):
            st.sidebar.error("GEMINI_API_KEY is not set in the .env file.")
        else:
            try:
                index_uploads(uploads)
                st.sidebar.success(st.session_state.status)
            except Exception as error:
                st.sidebar.error(f"Indexing failed: {error}")

    st.sidebar.info(st.session_state.status)
    st.session_state.use_rerank = st.sidebar.checkbox(
        "Optional rerank (extra Gemini call, slower)",
        value=False,
        help="Asks Gemini to reorder the retrieved chunks. A full project would call Cohere Rerank here.",
    )

    if st.sidebar.button("Clear documents and chat", use_container_width=True):
        st.session_state.chunks = []
        st.session_state.messages = []
        st.session_state.status = "No documents indexed yet."
        st.rerun()


def show_hits(hits: List[dict]) -> None:
    """Show the retrieved chunks so you can see why the answer looks the way it does."""
    with st.expander("Chunks used for this answer"):
        if not hits:
            st.write("No matching chunks.")
            return
        for hit in hits:
            st.markdown(f"**{hit['id']}**")
            st.caption(
                f"meaning score {hit['cosine']}  |  keyword score {hit['keyword_score']}  |  fused {hit['fusion']}"
            )
            st.write(hit["text"])


def draw_chat() -> None:
    st.title("Document RAG Assistant")
    st.caption("Upload a document, index it, then ask a question. The answer is written only from the retrieved chunks.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("hits"):
                show_hits(message["hits"])

    if not st.session_state.chunks:
        st.info("Index a document in the sidebar before asking a question. sample_notes.txt is in this folder.")

    question = st.chat_input(
        "Ask a question about your documents",
        disabled=not st.session_state.chunks,
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            # A follow-up like "when is it due?" needs the previous question too.
            search_text = question
            earlier_user = [m["content"] for m in st.session_state.messages[:-1] if m["role"] == "user"]
            if earlier_user:
                search_text = earlier_user[-1] + "\n" + question

            hits = search_chunks(search_text, st.session_state.chunks)
            if st.session_state.use_rerank:
                with st.spinner("Reranking chunks..."):
                    hits = rerank_hits(question, hits)

            context, used_hits = build_context(hits)
            if not context:
                answer = "I cannot find that in the uploaded documents."
                st.markdown(answer)
            else:
                history = st.session_state.messages[:-1]
                answer = st.write_stream(stream_answer(question, context, history))

            show_hits(used_hits)
        except Exception as error:
            answer = f"Something went wrong while answering: {error}"
            st.error(answer)
            used_hits = []

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": used_hits}
    )


def main() -> None:
    st.set_page_config(page_title="Document RAG Assistant", page_icon=":page_facing_up:", layout="wide")
    init_state()
    draw_sidebar()
    draw_chat()


def running_inside_streamlit() -> bool:
    """streamlit run sets this up. Plain python main.py does not."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if running_inside_streamlit():
        main()
    else:
        print("Start the page with this command, from this folder:")
        print(r"venv\Scripts\streamlit run main.py")
