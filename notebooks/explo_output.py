import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    df = pl.read_parquet("data/processed/*.parquet")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df, pl):
    df2 = df.with_columns(
        pl.col("arguments")
        .str.json_decode(
            dtype=pl.List(pl.Struct({"nom": pl.String, "resume": pl.String}))
        )
        .alias("arguments")
    ).select(
        [
            "doc_id",
            "publish_date",
            "is_nuclear",
            "chunks",
            "arguments",
            "retrieval_query",
        ]
    )
    return (df2,)


@app.cell
def _(df2):
    df2
    return


if __name__ == "__main__":
    app.run()
