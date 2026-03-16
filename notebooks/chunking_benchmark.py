import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import time
    import polars as pl
    import matplotlib.pyplot as plt

    return RecursiveCharacterTextSplitter, mo, pl, plt, time


@app.cell
def _(RecursiveCharacterTextSplitter, pl):
    # ---------------------------
    # parameters
    # ---------------------------

    RUNS = 1000
    CHUNK_SIZE = 1000
    OVERLAP = 200
    STEP = CHUNK_SIZE - OVERLAP

    df_raw = pl.read_excel("data/raw/senat_elec20_LLM_test.xlsx").rename(
        {"ID": "id"}
    )

    df = (
        df_raw.with_columns(
            pl.struct(["annee", "mois", "jour"])
            .rank("dense")
            .cast(pl.Int64)
            .alias("id_doc")
        )
        .drop("id", "part")
        .group_by("id_doc", maintain_order=True)
        .agg(pl.col("texte").str.join(" ").alias("full_text"), pl.all().first())
        .drop("texte")
    )

    texts = df["full_text"].to_list()


    # ---------------------------
    # Polars chunker
    # ---------------------------


    def polars_chunker(df: pl.DataFrame):

        return (
            df.with_columns(
                start=pl.int_ranges(0, pl.col("full_text").str.len_chars(), STEP)
            )
            .explode("start")
            .with_columns(
                chunk_text=pl.col("full_text").str.slice(
                    pl.col("start"), CHUNK_SIZE
                )
            )
            .select("chunk_text")
        )


    # ---------------------------
    # LangChain chunker
    # ---------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP,
        length_function=len,
    )


    def langchain_chunker(texts):

        chunks = []
        for t in texts:
            chunks.extend(splitter.split_text(t))

        return chunks

    return RUNS, df, langchain_chunker, polars_chunker, texts


@app.cell
def _(df, langchain_chunker, polars_chunker, texts):
    # Warm up run
    polars_chunker(df)
    langchain_chunker(texts)
    return


@app.cell
def _(RUNS, df, langchain_chunker, pl, plt, polars_chunker, texts, time):
    # ---------------------------
    # benchmarking
    # ---------------------------

    polars_times = []
    langchain_times = []

    for i in range(RUNS):
        # Polars
        start = time.perf_counter()
        polars_chunker(df)
        polars_times.append(time.perf_counter() - start)

        # LangChain
        start = time.perf_counter()
        langchain_chunker(texts)
        langchain_times.append(time.perf_counter() - start)


    # ---------------------------
    # results
    # ---------------------------

    results = pl.DataFrame(
        {"run": range(RUNS), "polars": polars_times, "langchain": langchain_times}
    )

    print(results.describe())


    # ---------------------------
    # plot
    # ---------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(results["run"], results["polars"], label="Polars")
    plt.plot(results["run"], results["langchain"], label="LangChain")

    plt.xlabel("Run")
    plt.ylabel("Runtime (seconds)")
    plt.title("Chunking Benchmark")
    plt.legend()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Polars is ~5 times faster than Langchain.
    From what I understand this is because RecursiveCharacterTextSplitter uses python 'for loops' internally where polars pipeline is fully vectorized ans works at the table level (vs document level for langchain).

    It's good practice to do a warm up run (we can see on the plot the spike for the first run for both)
    """)
    return


if __name__ == "__main__":
    app.run()
