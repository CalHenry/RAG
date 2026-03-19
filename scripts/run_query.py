from typing import Optional, cast

import typer

from rag.config import RETRIEVAL_QUERY
from rag.query.pipeline import run_pipeline

app = typer.Typer()


def range_callback(value: Optional[str]) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    try:
        start_str, end_str = value.split(":")
        start, end = int(start_str), int(end_str)
    except ValueError:
        raise typer.BadParameter("Use format START:END (e.g. 1:50)")
    if start > end:
        raise typer.BadParameter("Start must be <= end")
    return start, end


@app.command()
def main(
    ids: Optional[list[int]] = typer.Option(
        None,
        "--ids",
        help="Explicit doc_ids (e.g. --ids 1 --ids 5 --ids 6). Starts at 1.",
    ),
    range_: Optional[str] = typer.Option(
        None,
        "--range",
        callback=range_callback,
        help="Inclusive range START:END (e.g. --range 1:50)",
        is_eager=False,
    ),
    query: str = typer.Option(RETRIEVAL_QUERY, help="Retrieval query override"),
):
    if not ids and not range_:
        typer.echo("Error: provide either --ids or --range", err=True)
        raise typer.Exit(1)
    if ids and range_:
        typer.echo("Error: --ids and --range are mutually exclusive", err=True)
        raise typer.Exit(1)

    if range_:
        start, end = cast(tuple[int, int], range_)
        doc_ids = list(range(start, end + 1))
    else:
        assert ids is not None
        doc_ids = ids

    run_pipeline(doc_ids=doc_ids, retrieval_query=query)


if __name__ == "__main__":
    app()
