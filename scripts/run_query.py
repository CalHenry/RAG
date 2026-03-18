import typer

from rag.config import RETRIEVAL_QUERY
from rag.query.pipeline import run_pipeline

app = typer.Typer()


@app.command()
def main(
    ids: list[int] | None = typer.Option(
        None, help="Explicit doc_ids. Starts at 1. Can pass a range"
    ),
    range_: list[int] | None = typer.Option(
        None, "--range", help="Inclusive range [START END]. E.g. --range 1 50"
    ),
    query: str = typer.Option(RETRIEVAL_QUERY, help="Retrieval query override"),
):
    if not ids and not range_:
        typer.echo("Error: provide either --ids or --range", err=True)
        raise typer.Exit(1)
    if ids and range_:
        typer.echo("Error: --ids and --range are mutually exclusive", err=True)
        raise typer.Exit(1)

    doc_ids = ids if ids else list(range(range_[0], range_[1] + 1))
    run_pipeline(doc_ids=doc_ids, retrieval_query=query)


if __name__ == "__main__":
    app()
