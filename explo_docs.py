import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer, mo, pl


@app.cell
def _(pl):
    df_raw = pl.read_excel("data/raw/senat_elec20_LLM_test.xlsx").rename(
        {"ID": "id"}
    )
    return (df_raw,)


@app.cell
def _(df_raw):
    df_raw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Id var for each document + group by each document.
    Documents can be identified by the unique combo of 'annee', 'mois' and 'jour'.
    Documents are chunked into piece of max 20 000 charracters.
    We go from n rows of 20 000 characters texts to a single row of max.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Prepare data
    """)
    return


@app.cell
def _(df_raw, pl):
    # Create document id and aggregated at the document level
    df = (
        (
            df_raw.with_columns(
                pl.struct(["annee", "mois", "jour"])
                .rank("dense")
                .cast(pl.Int64)
                .alias("id_doc")
            )
            .drop("id", "part")
            .group_by("id_doc", maintain_order=True)
            .agg(
                pl.col("texte").str.join(" ").alias("full_text"), pl.all().first()
            )
            .drop("texte")
        )
        .with_columns(
            publish_date=pl.date(
                year=pl.col("annee"), month=pl.col("mois"), day=pl.col("jour")
            )
        )
        .drop(["annee", "mois", "jour"])
    )
    return (df,)


@app.cell
def _(df):
    df["full_text"].str.len_chars().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chunk with overlap
    """)
    return


@app.cell
def _(df, pl):
    CHUNK_SIZE = 1000
    OVERLAP = 200
    STEP = 100

    chunks = (
        (
            df.with_columns(text_len=pl.col("full_text").str.len_chars())
            .with_columns(
                start=pl.int_ranges(0, pl.col("text_len"), STEP)
            )  # generate sliding window start positions
            .explode("start")
            .with_columns(
                chunk_text=pl.col("full_text").str.slice(
                    pl.col("start"), CHUNK_SIZE
                )
            )  # slice the chunk
            .filter(
                pl.col("chunk_text").str.len_chars() > OVERLAP
            )  # remove tiny tail chunks if desired
            .drop(["full_text", "text_len", "start"])
        )
        .with_columns(chunk_text=pl.col("chunk_text").str.replace(r"\s+\S*$", ""))
        .with_columns(
            pl.col("chunk_text").cum_count().over("id_doc").alias("chunk_id")
        )
    )
    return (chunks,)


@app.cell
def _(chunks):
    chunks["chunk_text"].str.len_chars().describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Embedding
    """)
    return


@app.cell
def _(SentenceTransformer):
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    return (model,)


@app.cell
def _(chunks, model):
    texts = chunks["chunk_text"].to_list()
    embeddings = model.encode(texts, show_progress_bar=True)
    return


if __name__ == "__main__":
    app.run()
