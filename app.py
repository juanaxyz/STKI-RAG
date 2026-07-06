import streamlit as st
import os
import time
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from dotenv import load_dotenv
from google import genai

# ── Load API Key ──────────────────────────────────────────────
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(
    page_title="Semantic RAG Demo — Gemini 3.1 Flash-Lite", layout="wide"
)
st.title("Semantic Retrieval Augmented Generation (RAG) Demo")
st.caption("Powered by Gemini 3.1 Flash-Lite + gemini-embedding-001")


# ── Load Documents ────────────────────────────────────────────
def load_documents():
    docs = {}
    for fname in ["D1.md", "D2.md", "D3.md"]:
        path = os.path.join("dokumen", fname)
        with open(path, "r", encoding="utf-8") as f:
            docs[fname.replace(".md", "")] = f.read().strip()
    return docs


DOCS = load_documents()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API Key", value=GEMINI_API_KEY or "", type="password"
    )
    query = st.text_input("Search Query", value="What time do I leave for school?")
    run = st.button("Run Query", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Documents")
    for name, content in DOCS.items():
        with st.expander(name):
            st.markdown(content)

# ── Client ────────────────────────────────────────────────────
client = genai.Client(api_key=api_key) if api_key else None


# ── Helper: Embed ─────────────────────────────────────────────
def get_embedding(text: str) -> list[float]:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return resp.embeddings[0].values


# ── Helper: Cosine Similarity ─────────────────────────────────
def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ── Helper: Generate Answer + Citation (Gemini) ──────────────
def answer_query(
    query: str,
    best_key: str,
    scores: dict[str, float],
    docs: dict[str, str],
    metric: str = "cosine similarity",
) -> dict:
    doc_text = docs[best_key]
    scores_text = "\n".join([f"- {k}: {v:.4f}" for k, v in scores.items()])
    prompt = (
        f"You are a semantic Question Answering system.\n"
        f'A user asked: "{query}"\n\n'
        f"Based on {metric}, the best matching document is {best_key}.\n"
        f"Scores:\n{scores_text}\n\n"
        f"Content of {best_key}:\n{doc_text}\n\n"
        "Answer the query using ONLY the content above.\n"
        "Format your answer as:\n"
        f"ANSWER: [direct answer based on {best_key}]\n"
        "SNIPPET: [1 relevant sentence from the document]\n"
        "REASON: [why this document is the best match in 1 sentence]"
    )
    resp = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
    )
    lines = resp.text.strip().split("\n")
    result = {"answer": "", "snippet": "", "reason": ""}
    current = ""
    for line in lines:
        if line.startswith("ANSWER:"):
            current = "answer"
            result["answer"] = line.replace("ANSWER:", "").strip()
        elif line.startswith("SNIPPET:"):
            current = "snippet"
            result["snippet"] = line.replace("SNIPPET:", "").strip()
        elif line.startswith("REASON:"):
            current = "reason"
            result["reason"] = line.replace("REASON:", "").strip()
        elif current == "answer" and line.strip():
            result["answer"] += " " + line.strip()
        elif current == "reason" and line.strip():
            result["reason"] += " " + line.strip()
    return result


# ── Helper: Softmax → Probability Distribution ───────────────
def softmax(x: list[float]) -> np.ndarray:
    x = np.array(x)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# ── Helper: KL Divergence ────────────────────────────────────
def kl_divergence(p: list[float], q: list[float], eps: float = 1e-10) -> float:
    p, q = np.array(p) + eps, np.array(q) + eps
    return float(np.sum(p * np.log(p / q)))


# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Prompt-Based (Type A)",
        "Vector Search (Type B)",
        "KL-Divergence (Type C)",
        "Comparison",
    ]
)

# ══════════════════════════════════════════════════════════════
# TAB 1 — PROMPT-BASED RAG
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Type A: Prompt-Based (In-Context RAG)")
    st.markdown(
        "All documents are injected directly into the prompt. "
        "Gemini 3.1 Flash-Lite reads the full context and selects the most relevant one."
    )

    if run and api_key:
        st.divider()
        docs_text = "\n".join([f"- {k}: {v}" for k, v in DOCS.items()])
        prompt = (
            "You are a semantic Question Answering system.\n"
            "Given the following documents:\n\n"
            f"{docs_text}\n\n"
            f"USER QUERY: {query}\n\n"
            "RULES:\n"
            "1. Answer the query directly based on the most relevant document.\n"
            "2. Cite which document (D1, D2, or D3) you used.\n"
            "3. Include a brief relevant sentence from that document as a snippet.\n"
            "4. Explain why it's the best match.\n\n"
            "Format your answer as:\n"
            "ANSWER: [direct answer to the query]\n"
            "SOURCE: [D1/D2/D3]\n"
            "SNIPPET: [1 relevant sentence from the document]\n"
            "REASON: [why this document matches in 1 sentence]"
        )

        with st.spinner("Generating response..."):
            start_total = time.time()
            start = time.time()
            resp = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt,
            )
            elapsed = time.time() - start
            elapsed_total = time.time() - start_total

        lines = resp.text.strip().split("\n")
        answer = ""
        source = ""
        snippet = ""
        reason = ""
        current = ""
        for line in lines:
            if line.startswith("ANSWER:"):
                current = "answer"
                answer = line.replace("ANSWER:", "").strip()
            elif line.startswith("SOURCE:"):
                current = "source"
                source = line.replace("SOURCE:", "").strip()
            elif line.startswith("SNIPPET:"):
                current = "snippet"
                snippet = line.replace("SNIPPET:", "").strip()
            elif line.startswith("REASON:"):
                current = "reason"
                reason = line.replace("REASON:", "").strip()
            elif current == "answer" and line.strip():
                answer += " " + line.strip()
            elif current == "reason" and line.strip():
                reason += " " + line.strip()

        st.markdown(f"### Answer")
        st.success(answer)

        cols = st.columns(3)
        with cols[0]:
            st.info(f"**Source:** {source}")
        with cols[1]:
            st.info(f"**Snippet:** {snippet}")
        with cols[2]:
            st.info(f"**Reason:** {reason}")

        st.divider()
        st.markdown("### Execution Time")
        time_cols = st.columns(2)
        with time_cols[0]:
            st.metric("Response Time", f"{elapsed:.2f}s")
        with time_cols[1]:
            st.metric("Total Time", f"{elapsed_total:.2f}s")

    elif run and not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")

# ══════════════════════════════════════════════════════════════
# TAB 2 — VECTOR SEARCH RAG
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Type B: Vector Search (Embedding + Cosine Similarity)")
    st.markdown(
        "Documents are converted to 768-d vectors via gemini-embedding-001. "
        "The query vector is compared using cosine similarity."
    )

    if run and api_key:
        st.divider()
        start_total = time.time()

        # ── Step 1: Embed documents ──
        with st.spinner("Embedding documents..."):
            start_docs = time.time()
            doc_vecs = {}
            for key, text in DOCS.items():
                doc_vecs[key] = get_embedding(text)
            elapsed_docs = time.time() - start_docs

        # ── Step 2: Embed query ──
        with st.spinner("Embedding query..."):
            start_query = time.time()
            q_vec = get_embedding(query)
            elapsed_query = time.time() - start_query

        # ── Step 3: Cosine Similarity ──
        scores = {k: cosine_sim(q_vec, v) for k, v in doc_vecs.items()}
        best_key = max(scores, key=scores.get)

        # ── Step 4: Generate answer + citation ──
        with st.spinner("Generating answer..."):
            start_answer = time.time()
            ans = answer_query(query, best_key, scores, DOCS)
            elapsed_answer = time.time() - start_answer

        elapsed_total = time.time() - start_total

        # ── Display Results ──
        st.markdown("### Answer")
        st.success(ans["answer"])

        cols = st.columns(3)
        with cols[0]:
            st.info(f"**Source:** {best_key} (score: {scores[best_key]:.4f})")
        with cols[1]:
            st.info(f"**Snippet:** {ans['snippet']}")
        with cols[2]:
            st.info(f"**Reason:** {ans['reason']}")

        st.divider()
        st.markdown("### Execution Time")
        time_cols = st.columns(4)
        with time_cols[0]:
            st.metric("Embed Docs", f"{elapsed_docs:.2f}s")
        with time_cols[1]:
            st.metric("Embed Query", f"{elapsed_query:.2f}s")
        with time_cols[2]:
            st.metric("Generate Answer", f"{elapsed_answer:.2f}s")
        with time_cols[3]:
            st.metric(
                "Total Time",
                f"{elapsed_total:.2f}s",
                delta=f"-{(elapsed_total - (elapsed_docs + elapsed_query + elapsed_answer)):.2f}s",
            )

        with st.expander("Similarity Scores"):
            for k, v in scores.items():
                bar_color = "green" if k == best_key else "gray"
                st.markdown(f"**{k}**: {v:.4f}")
                st.progress(v)

        # ── Step 5: 3D Visualization ──
        st.divider()
        st.markdown("### 3D Vector Space Visualization")
        st.caption("Drag to rotate, scroll to zoom, hover for details")

        all_vecs = list(doc_vecs.values()) + [q_vec]
        pca = PCA(n_components=3)
        reduced = pca.fit_transform(all_vecs)

        doc_coords = reduced[:3]
        q_coord = reduced[3]

        labels = list(DOCS.keys()) + ["QUERY"]
        colors = [
            "rgba(0,200,0,1)" if k == best_key else "rgba(150,150,150,1)"
            for k in DOCS.keys()
        ] + ["rgba(255,0,0,1)"]
        sizes = [14, 14, 14, 20]

        hover_texts = [f"<b>{k}</b><br>Score: {scores[k]:.4f}" for k in DOCS.keys()] + [
            f"<b>QUERY</b><br>{query}"
        ]

        fig = go.Figure()

        # Document points
        fig.add_trace(
            go.Scatter3d(
                x=doc_coords[:, 0],
                y=doc_coords[:, 1],
                z=doc_coords[:, 2],
                mode="markers+text",
                marker=dict(size=sizes[:3], color=colors[:3]),
                text=list(DOCS.keys()),
                textposition="top center",
                hovertext=hover_texts[:3],
                hoverinfo="text",
                name="Documents",
            )
        )

        # Query point
        fig.add_trace(
            go.Scatter3d(
                x=[q_coord[0]],
                y=[q_coord[1]],
                z=[q_coord[2]],
                mode="markers+text",
                marker=dict(size=[20], color=["rgba(255,0,0,1)"], symbol="diamond"),
                text=["QUERY"],
                textposition="top center",
                hovertext=[f"<b>QUERY</b><br>{query}"],
                hoverinfo="text",
                name="Query",
            )
        )

        # Lines from query to each document
        for i, key in enumerate(DOCS.keys()):
            opacity = 0.9 if key == best_key else 0.3
            width = 4 if key == best_key else 1
            fig.add_trace(
                go.Scatter3d(
                    x=[q_coord[0], doc_coords[i, 0]],
                    y=[q_coord[1], doc_coords[i, 1]],
                    z=[q_coord[2], doc_coords[i, 2]],
                    mode="lines",
                    line=dict(width=width, color="rgba(100,100,255,1)"),
                    opacity=opacity,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        fig.update_layout(
            scene=dict(
                xaxis_title="PC1",
                yaxis_title="PC2",
                zaxis_title="PC3",
                bgcolor="rgba(240,240,240,1)",
            ),
            height=550,
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05),
        )
        st.plotly_chart(fig, use_container_width=True)

    elif run and not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")

# ══════════════════════════════════════════════════════════════
# TAB 3 — KL-DIVERGENCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Type C: KL-Divergence (Probabilistic Distribution)")
    st.markdown(
        "Vectors from gemini-embedding-001 are converted to probability distributions "
        "(via softmax). KL-Divergence measures the information distance: "
        "**smaller = more similar**."
    )

    if run and api_key:
        st.divider()
        start_total = time.time()

        # ── Step 1: Embed documents ──
        with st.spinner("Embedding documents..."):
            start_docs = time.time()
            doc_vecs = {}
            for key, text in DOCS.items():
                doc_vecs[key] = get_embedding(text)
            elapsed_docs = time.time() - start_docs

        # ── Step 2: Embed query & softmax ──
        with st.spinner("Embedding query..."):
            start_query = time.time()
            q_vec = get_embedding(query)
            elapsed_query = time.time() - start_query

        # ── Step 3: Softmax → probability distributions ──
        q_prob = softmax(q_vec)
        doc_probs = {k: softmax(v) for k, v in doc_vecs.items()}

        # ── Step 4: KL-Divergence ──
        scores = {k: kl_divergence(q_prob, v) for k, v in doc_probs.items()}
        best_key = min(scores, key=scores.get)

        # ── Step 5: Generate answer + citation ──
        with st.spinner("Generating answer..."):
            start_answer = time.time()
            ans = answer_query(query, best_key, scores, DOCS, metric="KL-Divergence")
            elapsed_answer = time.time() - start_answer

        elapsed_total = time.time() - start_total

        # ── Display Results ──
        st.markdown("### Answer")
        st.success(ans["answer"])

        cols = st.columns(3)
        with cols[0]:
            st.info(f"**Source:** {best_key} (D_KL: {scores[best_key]:.4f})")
        with cols[1]:
            st.info(f"**Snippet:** {ans['snippet']}")
        with cols[2]:
            st.info(f"**Reason:** {ans['reason']}")

        st.divider()
        st.markdown("### Execution Time")
        time_cols = st.columns(4)
        with time_cols[0]:
            st.metric("Embed Docs", f"{elapsed_docs:.2f}s")
        with time_cols[1]:
            st.metric("Embed Query", f"{elapsed_query:.2f}s")
        with time_cols[2]:
            st.metric("Generate Answer", f"{elapsed_answer:.2f}s")
        with time_cols[3]:
            st.metric(
                "Total Time",
                f"{elapsed_total:.2f}s",
                delta=f"-{(elapsed_total - (elapsed_docs + elapsed_query + elapsed_answer)):.2f}s",
            )

        with st.expander("KL-Divergence Scores (lower = better)"):
            scores_list = list(scores.values())
            min_s, max_s = min(scores_list), max(scores_list)
            for k, v in scores.items():
                norm = (max_s - v) / (max_s - min_s) if max_s != min_s else 1.0
                st.markdown(f"**{k}**: {v:.4f}")
                st.progress(norm)

    elif run and not api_key:
        st.error("Please enter your Gemini API Key in the sidebar.")

# ══════════════════════════════════════════════════════════════
# TAB 4 — COMPARISON
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Comparison: All 3 Methods")
    st.markdown(
        "Side-by-side comparison of all retrieval approaches on the same query."
    )

    st.divider()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Type A — Prompt-Based")
        st.markdown(
            """
            | Dimension | Detail |
            |-----------|--------|
            | **Logic** | Generative Reasoning (LLM reads all text) |
            | **Tokens** | Most (all docs sent every query) |
            | **Speed** | Slowest (large prompt) |
            | **Output** | Answer + Source + Snippet |
            | **Dependencies** | Gemini 3.1 Flash-Lite only |
            """
        )

    with col2:
        st.markdown("### Type B — Vector Search")
        st.markdown(
            """
            | Dimension | Detail |
            |-----------|--------|
            | **Logic** | Cosine Similarity (geometric) |
            | **Tokens** | Medium (embed once) |
            | **Speed** | Fast (math-based) |
            | **Output** | Answer + Source + Snippet |
            | **Dependencies** | gemini-embedding-001 + Gemini Flash-Lite |
            """
        )

    with col3:
        st.markdown("### Type C — KL-Divergence")
        st.markdown(
            """
            | Dimension | Detail |
            |-----------|--------|
            | **Logic** | KL Divergence (probabilistic) |
            | **Tokens** | Medium (embed once) |
            | **Speed** | Fast (math-based) |
            | **Output** | Answer + Source + Snippet |
            | **Dependencies** | gemini-embedding-001 + Gemini Flash-Lite |
            """
        )

    st.divider()
    st.markdown("### Key Insight")
    st.info(
        "**Type A (Prompt)** and **Type B (Vector Search)** both identify **D1** "
        'as the most relevant document for "What time do I leave for school?" '
        "— despite no exact keyword match. This demonstrates **semantic understanding** "
        "over traditional keyword-based methods."
    )

    st.markdown("### Formula Comparison")
    st.code(
        "Type A: LLM reasoning over full document text\n"
        "Type B: cos(θ) = (A·B) / (||A|| × ||B||)\n"
        "Type C: D_KL(P‖Q) = Σ P(i) × log(P(i) / Q(i))",
        language=None,
    )
