from pathlib import Path

import polars as pl

from rag.config import AI_OUTPUT_PATH, FINAL_OUTPUT_PATH, VECTOR_OUTPUT_PATH


def merge_parts(
    part1_path: str | Path, part2_path: str | Path, output_path: str | Path
) -> None:
    df1 = pl.read_parquet(part1_path)
    df2 = pl.read_parquet(part2_path)
    merged = df1.join(df2, on="doc_id", how="inner").sort("doc_id")

    df2 = merged.with_columns(
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
            "queried_at",
            "confidence",
        ]
    )

    merged.write_parquet(output_path)
    print(f"Merged {len(merged)} rows → {output_path}")


def main():
    merge_parts(VECTOR_OUTPUT_PATH, AI_OUTPUT_PATH, FINAL_OUTPUT_PATH)


if __name__ == "__main__":
    main()
