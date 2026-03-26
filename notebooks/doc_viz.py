# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ipython==9.11.0",
#     "marimo>=0.21.1",
#     "matplotlib==3.10.8",
#     "numpy==2.4.3",
#     "polars==1.39.3",
#     "rag",
#     "sentence-transformers==5.3.0",
# ]
#
# [tool.uv.sources]
# rag = { git = "https://github.com/CalHenry/RAG.git" }
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _(user_query):
    user_query
    return


@app.cell(hide_code=True)
def _(document_selector):
    document_selector
    return


@app.cell
def _(fig, mo, smooth_switch):
    # Plot
    # Mak the plot reactive - User can select obs using the mouse on the plot
    ax = mo.ui.matplotlib(fig.gca())
    ax

    mo.vstack([ax, smooth_switch])
    return (ax,)


@app.cell
def _(ax, chunk_id, df, mo, pl, similarity):
    # Show dataset
    mask = ax.value.get_mask(chunk_id, similarity)

    # 2 tab: 1 for full dataset, one for the subset from the reactive plot
    tabs = mo.ui.tabs(
        {
            "Full dataset": df,
            "Selection": df.with_columns(pl.col("chunk_id")).filter(mask),
        }
    )
    tabs
    return (mask,)


@app.cell
def _(chunk_selector):
    chunk_selector
    return


@app.cell
def _(chunk_display):
    chunk_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    # UI elements
    return


@app.cell
def _(mo):
    user_query = mo.ui.text_area(
        value="énergie nucléaire, centrale nucléaire, réacteur atomique",
        placeholder="Please enter text",
        label="Search query: ",
    )
    return (user_query,)


@app.cell
def _(mo, np):
    # Documents dropdown
    docs = np.sort(np.arange(1, 45))  # there are 45 docs total
    document_selector = mo.ui.dropdown(
        options={f"Doc {doc}": doc for doc in docs},
        label="Select a document: ",
        value="Doc 1",
    )
    return (document_selector,)


@app.cell
def _(ax, df, mask, mo, pl):
    # Chunks dropdown
    # Show the full dataset or the if there is a selection
    source_df = df.with_columns(pl.col("_distance").sort(descending=False))

    has_selection = type(ax.value).__name__ == "BoxSelection"
    top_k_chunks = (
        source_df.filter(mask).head(15) if has_selection else source_df.head(15)
    )

    chunk_selector = mo.ui.dropdown(
        options={
            f"Chunk {row['chunk_id']}": row["chunk_text"]
            for row in top_k_chunks.iter_rows(named=True)
        },
        label="Select a chunk: ",
    )
    return (chunk_selector,)


@app.cell
def _(chunk_selector, format_sentences, mo):
    # display the selected text, reacts automatically
    chunk_display = mo.md(f"""
    ### {chunk_selector.selected_key}:

    {format_sentences(chunk_selector.value) or "*Select a chunk above*"}
    """)
    return (chunk_display,)


@app.cell
def _(mo):
    smooth_switch = mo.ui.switch(label="Smoothing", value=True)
    return (smooth_switch,)


@app.cell
def _():
    # Execution of the function
    return


@app.cell
async def _(
    document_relevance_map,
    document_selector,
    embedder,
    pl,
    smooth_switch,
    user_query,
):
    df, fig, chunk_id, similarity = await document_relevance_map(
        embedder,
        user_query.value,
        doc_id=document_selector.value,
        smoothing=smooth_switch.value,
    )

    df = df.sort(pl.col("_distance"))
    return chunk_id, df, fig, similarity


@app.cell
def _():
    # Code
    return


@app.cell
def _():
    import os
    import re
    from pathlib import Path

    import lancedb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.ndimage import gaussian_filter1d
    from sentence_transformers import SentenceTransformer

    from rag.config import MODEL_NAME, MODEL_PATH
    from rag.data_models import ChunkResult, RAGDeps
    from rag.query.helpers import retrieve

    return (
        MODEL_NAME,
        MODEL_PATH,
        RAGDeps,
        SentenceTransformer,
        gaussian_filter1d,
        mo,
        np,
        pl,
        plt,
        re,
        retrieve,
    )


@app.cell
def _(MODEL_NAME, MODEL_PATH, SentenceTransformer):
    # Embedder setup ------------------------------------------------------------
    use_local = False  # pull from HF
    embedder = SentenceTransformer(
        MODEL_PATH if use_local else MODEL_NAME, local_files_only=use_local
    )
    return (embedder,)


@app.cell
def _(RAGDeps, SentenceTransformer, gaussian_filter1d, np, pl, plt, retrieve):
    async def document_relevance_map(
        embedder: SentenceTransformer,
        retrieval_query: str,
        doc_id: int,
        smoothing: bool = True,
        smooth_sigma: int = 8,
        top_k_markers: int = 5,
    ):
        """
        Dataviz: Distribution of the' _distance' variable for a document of the vector database.
        Steps:
            - set the top_k to > 1000 to be sure to retrieve all the chunks of the document
            - retrieve all the ChunkResult (chunk_id, chunk_text, publish_date, _distance) for a document_id
            - convert to a dataframe
            - normalize '_distance', and smooth it using gaussian kernel (can be disable with 'smoothing' parameter)
            - plot the distribution, red dots for the top 5 chunks
        """

        # override top_k to get all chunks

        deps = RAGDeps.create(
            embedder=embedder,
            retrieval_query=retrieval_query,
            doc_id=doc_id,
        )
        deps.top_k = 10_000

        # get the chunks and _distance
        chunks = await retrieve(deps, retrieval_query, doc_id)

        df = pl.DataFrame(chunks).sort("chunk_id")

        distances = df["_distance"].to_numpy()
        similarity = 1 - (distances / distances.max())  # normalized 0–1

        if smoothing:
            similarity = gaussian_filter1d(similarity, sigma=smooth_sigma)
        chunk_ids = df["chunk_id"].to_numpy()

        # Top-k positions (lowest distance = most relevant)
        top_k_idx = np.argsort(distances)[:top_k_markers]

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(chunk_ids, similarity, alpha=0.3, color="steelblue")
        ax.plot(chunk_ids, similarity, color="steelblue", linewidth=1.5)
        ax.scatter(
            chunk_ids[top_k_idx],
            similarity[top_k_idx],
            color="red",
            zorder=5,
            s=60,
            label=f"Top-{top_k_markers} chunks",
        )

        ax.set_xlabel("Chunk position in document")
        ax.set_ylabel("Relevance score")
        ax.set_title(f"Argument map — doc {doc_id} | query: '{retrieval_query}'")
        ax.legend()
        plt.tight_layout()
        # plt.show()

        return df, fig, chunk_ids, similarity

    return (document_relevance_map,)


@app.cell
def _(re):
    ABBREVIATIONS = {
        "e.g.",
        "i.e.",
        "mr.",
        "mrs.",
        "dr.",
        "vs.",
        "etc.",
        "u.s.",
        "fig.",
    }


    def is_abbreviation(text, start):
        # Look backwards a bit to check for abbreviation
        snippet = text[max(0, start - 10) : start + 1].lower()
        return any(snippet.endswith(abbr) for abbr in ABBREVIATIONS)


    def format_sentences(text: str, min_chars: int = 10) -> str:
        if not text:
            return "*Select a chunk above*"

        result = []
        last_break = 0

        for match in re.finditer(r"[.!?]\s+", text):
            start, end = match.span()

            # Skip decimals (digit . digit)
            if start > 0 and end < len(text):
                if text[start - 1].isdigit() and text[end].isdigit():
                    continue

            # Skip abbreviations
            if is_abbreviation(text, start):
                continue

            # Check next char is uppercase (likely sentence start)
            if end < len(text) and not text[end].isupper():
                continue

            current_len = start - last_break

            # Estimate next sentence length
            next_chunk = text[end:]
            next_len = len(re.split(r"[.!?\n]", next_chunk)[0])

            if current_len >= min_chars and next_len >= min_chars:
                result.append(text[last_break:end].strip())
                last_break = end

        result.append(text[last_break:].strip())

        formatted = "  \n".join(result)

        # Escape ordered list markdown issues
        formatted = re.sub(r"^(\d+)\.\s", r"\1\\. ", formatted, flags=re.MULTILINE)

        return formatted

    return (format_sentences,)


if __name__ == "__main__":
    app.run()
