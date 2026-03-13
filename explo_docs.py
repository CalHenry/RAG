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
    df = pl.read_excel('data/raw/senat_elec20_LLM_test.xlsx')
    return (df,)


@app.cell
def _(df):
    df.describe()
    return


@app.cell
def _(df):
    df['texte'].str.len_chars()
    return


if __name__ == "__main__":
    app.run()
