import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return mo, pl


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


@app.cell
def _(df_raw, pl):
    # Create document id and aggregated at the document level
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
    return (df,)


@app.cell
def _(df):
    df["full_text"].str.len_chars().describe()
    return


if __name__ == "__main__":
    app.run()
