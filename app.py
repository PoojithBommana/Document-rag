
import hashlib
import html

import streamlit as st

from main import answer_question, build_vectorstore


st.set_page_config(
    page_title="DocuMind",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg: #101522;
        --sidebar: #0B1120;
        --panel: #171F30;
        --panel-hover: #202B40;
        --border: #2A3549;
        --text: #EDF2FA;
        --muted: #9AA8BD;
        --accent: #9585FF;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(16, 21, 34, 0.95);
    }

    [data-testid="stSidebar"] {
        background: var(--sidebar);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] * {
        color: #E1E8F5;
    }

    .brand {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.8px;
        color: #FFFFFF;
    }

    .brand span {
        color: var(--accent);
    }

    .muted {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.65;
    }

    .workspace-heading {
        font-size: 1.9rem;
        letter-spacing: -0.8px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 5px;
    }

    .workspace-subtitle {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 24px;
    }

    .stat-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px 17px;
        min-height: 83px;
    }

    .stat-label {
        color: var(--muted);
        font-size: 0.78rem;
        margin-bottom: 7px;
    }

    .stat-value {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 700;
    }

    /* Make user questions impossible to confuse with answers. */
    .question-card {
        background: #252347;
        border: 1px solid #51458A;
        border-left: 4px solid var(--accent);
        border-radius: 12px;
        padding: 17px 19px;
        margin: 12px 0 14px 0;
        overflow-wrap: anywhere;
    }

    .question-label {
        color: #B8ADFF;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 9px;
    }

    .question-text {
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 500;
        line-height: 1.8;
        white-space: pre-wrap;
    }

    .answer-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .answer-label {
        color: #9C8FFF;
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .answer-card p, .answer-card li {
        color: #E1E8F2;
        line-height: 1.85;
    }

    .answer-card h1, .answer-card h2, .answer-card h3 {
        color: #FFFFFF;
    }

    .answer-card code {
        color: #D5CCFF;
        background: #252D40;
        border-radius: 4px;
        padding: 2px 5px;
    }

    [data-testid="stExpander"] {
        background: #141C2A;
        border: 1px solid var(--border);
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p {
        color: #DCE5F3 !important;
    }

    [data-testid="stFileUploader"] {
        background: #141C2A;
        border: 1px dashed #46516C;
        border-radius: 11px;
        padding: 9px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small {
        color: #DCE5F3;
    }

    .stTextInput input,
    .stTextArea textarea,
    [data-testid="stChatInput"] textarea {
        background: #1A2334 !important;
        color: #FFFFFF !important;
        border: 1px solid #3B4760 !important;
        border-radius: 12px !important;
    }

    [data-testid="stChatInput"] {
        background: var(--bg);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #A0AEC0 !important;
        opacity: 1;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        border-radius: 9px;
        min-height: 40px;
        font-weight: 600;
        border: 1px solid #3B4760;
    }

    .stButton > button[kind="primary"] {
        background: #7564EA;
        color: #FFFFFF;
        border-color: #7564EA;
    }

    .stButton > button[kind="primary"]:hover {
        background: #8777FA;
        border-color: #8777FA;
    }

    [data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: var(--border);
    }

    .empty-state {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 18px;
        text-align: center;
        padding: 42px 22px;
        margin-top: 28px;
    }

    .empty-icon {
        font-size: 40px;
        margin-bottom: 12px;
    }

    .empty-title {
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .empty-copy {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.8;
        max-width: 440px;
        margin: auto;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div class="brand">📚 Docu<span>Mind</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="muted">Your AI document workspace</p>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Knowledge base")
    st.caption("Upload PDFs to start asking questions.")

    uploaded_files = st.file_uploader(
        "Add PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files.",
    )

    signature = None

    if uploaded_files:
        signature = tuple(
            (
                file.name,
                file.size,
                hashlib.sha256(file.getvalue()).hexdigest(),
            )
            for file in uploaded_files
        )

        st.markdown("**Selected documents**")

        for file in uploaded_files:
            st.markdown(
                f"📄 {html.escape(file.name)}",
                unsafe_allow_html=True,
            )
            st.caption(f"{file.size / 1024:.1f} KB")

        is_new_upload = (
            signature != st.session_state.uploaded_signature
        )

        if is_new_upload:
            st.info("These files haven't been indexed yet.")

            if st.button(
                "Build knowledge base",
                type="primary",
                use_container_width=True,
            ):
                try:
                    with st.spinner(
                        "Extracting text and building the index..."
                    ):
                        pdf_data = [
                            (file.name, file.getvalue())
                            for file in uploaded_files
                        ]
                        new_vectorstore = build_vectorstore(pdf_data)

                    # Replace the old index only after success.
                    st.session_state.vectorstore = new_vectorstore
                    st.session_state.uploaded_signature = signature
                    st.session_state.chat_history = []

                    st.rerun()

                except Exception as exc:
                    st.error(f"Processing failed: {exc}")

        else:
            st.success("Knowledge base is ready.")

    elif st.session_state.vectorstore is not None:
        st.success("Previously indexed documents are ready.")
        st.caption(
            "Removing files from this picker does not delete "
            "the existing index."
        )

    else:
        st.caption("No documents indexed yet.")

    st.divider()
    st.markdown("### Workspace")

    if st.button("＋ New conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.session_state.chat_history:
        st.caption(
            f"{len(st.session_state.chat_history)} questions "
            "in this conversation"
        )

    st.divider()
    st.markdown(
        """
        <div class="muted">
            <b>How it works</b><br><br>
            01 &nbsp; Upload PDFs<br><br>
            02 &nbsp; Create searchable embeddings<br><br>
            03 &nbsp; Retrieve and rerank passages<br><br>
            04 &nbsp; Generate cited answers
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Main workspace header
# --------------------------------------------------

st.markdown(
    """
    <div class="workspace-heading">Document assistant</div>
    <div class="workspace-subtitle">
        Explore your PDFs. Get answers grounded in your documents.
    </div>
    """,
    unsafe_allow_html=True,
)

# Summary cards
col1, col2, col3 = st.columns(3)

document_count = len(st.session_state.uploaded_signature or ())
question_count = len(st.session_state.chat_history)
ready = st.session_state.vectorstore is not None

stats = [
    ("INDEXED DOCUMENTS", str(document_count)),
    ("QUESTIONS ASKED", str(question_count)),
    ("KNOWLEDGE BASE", "Ready" if ready else "Not ready"),
]

for column, (label, value) in zip([col1, col2, col3], stats):
    with column:
        st.markdown(
            f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("")


# --------------------------------------------------
# Empty state
# --------------------------------------------------

if not ready:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✦</div>
            <div class="empty-title">Your documents, one conversation away</div>
            <div class="empty-copy">
                Upload your PDFs in the sidebar and build your
                knowledge base. Then ask questions and explore
                answers with document-level citations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Conversation
# --------------------------------------------------

else:
    st.markdown("### Conversation")

    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-icon">💬</div>
                <div class="empty-title">What would you like to understand?</div>
                <div class="empty-copy">
                    Ask about a concept, summarize a document,
                    compare ideas, or find specific information
                    across your uploaded PDFs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Try asking")

        suggestions = [
            "Summarize the main points",
            "Explain a key concept",
            "What are the important findings?",
        ]

        suggestion_cols = st.columns(3)

        for col, suggestion in zip(suggestion_cols, suggestions):
            with col:
                if st.button(suggestion, use_container_width=True):
                    st.session_state.pending_question = suggestion
                    st.rerun()

    # Display each question explicitly, not in default chat bubbles.
    for index, item in enumerate(st.session_state.chat_history):
        question_html = html.escape(item["question"])

        st.markdown(
            f"""
            <div class="question-card">
                <div class="question-label">YOUR QUESTION · {index + 1}</div>
                <div class="question-text">{question_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="answer-card"><div class="answer-label">'
            '✦ &nbsp; DOCUMIND ANSWER</div>',
            unsafe_allow_html=True,
        )

        # Render Markdown normally so code blocks, lists, and headings work.
        st.markdown(item["answer"])

        st.markdown("</div>", unsafe_allow_html=True)

        sources = item.get("sources", [])

        with st.expander(f"📎 Sources and citations · {len(sources)}"):
            if not sources:
                st.caption("No source passages were retrieved.")

            for source_index, source in enumerate(sources, start=1):
                st.markdown(
                    f"**{source_index}. "
                    f"{html.escape(str(source['filename']))}**"
                    f" · Page {source['page']}"
                )

                st.caption(
                    f"Rerank score: {source['rerank_score']}"
                )
                st.text(source["excerpt"])

                if source_index < len(sources):
                    st.divider()

    # --------------------------------------------------
    # Persistent chat input
    # --------------------------------------------------

    pending = st.session_state.pop("pending_question", "")

    question = st.chat_input(
        "Ask anything about your documents...",
        max_chars=2000,
    )

    # Suggested questions are processed through the same backend.
    if pending:
        question = pending

    if question is not None:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner(
                    "Searching documents, reranking passages, "
                    "and generating your answer..."
                ):
                    result = answer_question(
                        st.session_state.vectorstore,
                        question.strip(),
                    )

                st.session_state.chat_history.append(
                    {
                        "question": question.strip(),
                        "answer": result["answer"],
                        "sources": result["sources"],
                    }
                )

                st.rerun()

            except Exception as exc:
                st.error(f"Could not answer your question: {exc}")