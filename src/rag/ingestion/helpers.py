import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer

"""
Steps:
    1. Prepare the raw data for chunking with overlap
    2. Chunk the documents
    3. Embedd the chunks
    4. Create and store the chunks in the vector database
"""


# 1. Prepare raw data ---------------
# Create document id and aggregate at the document level
def prepare_raw_data(lf_raw: pl.LazyFrame) -> pl.LazyFrame:
    """
    Input data are documents in chunks of ~20K characters + date data. We first need to reconstruct the documents (documents level).
    Steps:
        - create an ID for each doc based on the 3 date variables ('annee', 'mois' and 'jour')
        - group by this new ID and join the chunks. Date values are the same for all the chunks so we keep the first value
        - aggregate the date informations into one Date variable
    """
    lf = (
        (
            lf_raw.with_columns(
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
        .with_columns(
            publish_date=pl.date(
                year=pl.col("annee"), month=pl.col("mois"), day=pl.col("jour")
            )
        )
        .drop(["annee", "mois", "jour"])
    )

    return lf


# 2. Chunk the documents ---------------
def chunk_documents(
    lf: pl.LazyFrame, chunk_size: int, overlap: int, step: int
) -> pl.LazyFrame:
    """
    Chunk the documents with overlap before the embedding (done in polars for better performances).

    Steps:
        - compute the character length of each document
        - create a sliding window based on the value of step. This computes the cut points.
        - explode based on the cut point. This duplicates the text var into n=step rows with the value of the new var "start" being 0 + step*n (ex: [100, 200, 300,...]). Required to cut correctly with overlap.
        - slice the texts into the chunks based on the start index given by "start".
        Overlap is determined by step and chunk_size: step < chunk_size means windows overlap. Each chunk share (chunk_size - step) characters in common.
        - remove the tiny tail chunk if this chunk is < to the overlap (100% of its content overlaps with the previous chunk, thus no new infos)
        - remove the 3 vars used to create the chunks
        - clean the edges of the chunks. The cuts  are based on character and can happend mid-word. For RAG we don't want partial words.
        We used regex to match the last word sequence followed by a whitespace. (ex: 'hello world exam' will become 'hellow world')
        - add an ID var for the chunks by documents (for each document the chunk id start at 1). doc_id + chunk_id = unique identifier

    Mental model:
        lf (documents)
          → add text length          # measure
          → generate start positions # plan cut points
          → explode                  # one row per cut point
          → slice chunks             # actually cut
          → filter tiny tails        # discard noise
          → drop scaffolding         # clean up
          → trim partial words       # clean boundaries
          → number chunks per doc    # addressability
    """
    lf_chunks = (
        (
            lf.with_columns(text_len=pl.col("full_text").str.len_chars())
            .with_columns(
                start=pl.int_ranges(0, pl.col("text_len"), step)
            )  # generate sliding window start positions
            .explode("start")
            .with_columns(
                chunk_text=pl.col("full_text").str.slice(pl.col("start"), chunk_size)
            )  # slice the chunk
            .filter(
                pl.col("chunk_text").str.len_chars() > overlap
            )  # remove tiny tail chunk
            .drop(["full_text", "text_len", "start"])
        )
        .with_columns(chunk_text=pl.col("chunk_text").str.replace(r"\s+\S*$", ""))
        .with_columns(pl.col("chunk_text").cum_count().over("id_doc").alias("chunk_id"))
    )

    return lf_chunks


# 3. Embedding ---------------
def embedd(
    model: SentenceTransformer, chunks_lf: pl.LazyFrame, chunk_column: str
) -> pl.DataFrame:
    """
    Embedd the chunks using a SentenceTransformer model.
    Steps:
        - select and convert the chunks to a list
        - embedd with the model
        - add the embeddings to the lazyframe as a list of float32 then collect.
        This dataframe will can be send directly to the vector database (lancedb accepts polars natively trought Arrow)
    """
    texts = chunks_lf.select(chunk_column).collect().to_series().to_list()
    embeddings = model.encode(texts, show_progress_bar=True)
    df_embeddings = chunks_lf.with_columns(
        pl.Series(
            "vector",
            embeddings.astype(np.float32).tolist(),  # float32 for the vector database
            dtype=pl.List(pl.Float32),
        )
    ).collect()

    return df_embeddings
