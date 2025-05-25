import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_semantic_search as aux_semantics
import bm25s

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    doc: str
    label: str
    query: str


def evaluate_vsm_queries(
    paths: Dict[str, Path],
    documents: List[DocumentSection],
    top_k: int = 20,
    use_multivector: bool = True,
    use_expansion: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs VSM retrieval on each DocumentSection in `documents`, builds a per-query
    DataFrame of (doc_id, score, label) indexed by rank, and then returns:
      - summary_df: DataFrame with one row per query, containing
          doc, label, label_count_top5/10/20, matched_rank, matched_score
      - results_map: dict mapping each doc.doc → its full ranked DataFrame
    """
    # 1) ONE‑TIME LOAD of heavy resources
    resources = aux_vsm.load_word2vec_resources(paths, use_multivector=use_multivector)

    results_map: Dict[str, pd.DataFrame] = {}
    summary = []

    for doc in documents:
        q = doc.query
        logging.info(f"Testing query: {q!r} at top_k={top_k}")
        out = aux_vsm.run_word2vec_query_preloaded(
            resources,
            query=q,
            top_k=top_k,
            use_expansion=use_expansion
        )

        # 2) Build the per-query DataFrame
        raw = out["results"]
        df = (
            pd.DataFrame(raw)
              .loc[:, ['rank', 'doc_id', 'score', 'grandparent']]
              .set_index('rank')
              .rename(columns={'grandparent': 'label'})
        )
        results_map[doc.doc] = df

        # 3) Compute summary metrics
        count5  = df.head(5) ['label'].eq(doc.label).sum()
        count10 = df.head(10)['label'].eq(doc.label).sum()
        count20 = df.head(20)['label'].eq(doc.label).sum()

        match = df[df['doc_id'] == doc.doc]
        if not match.empty:
            matched_rank  = int(match.index[0])
            matched_score = float(match['score'].iloc[0])
        else:
            matched_rank  = None
            matched_score = None

        summary.append({
            'doc':                doc.doc,
            'label':              doc.label,
            'label_count_top5':   count5,
            'label_count_top10':  count10,
            'label_count_top20':  count20,
            'matched_rank':       matched_rank,
            'matched_score':      matched_score
        })

    # 4) Build summary DataFrame
    summary_df = pd.DataFrame(summary)
    return summary_df, results_map

def evaluate_bm25_queries(
    paths: Dict[str, Path],
    documents: List[DocumentSection],
    top_k: int = 20,
    use_multivector: bool = True,
    use_expansion: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Runs VSM retrieval on each DocumentSection in `documents`, builds a per-query
    DataFrame of (doc_id, score, label) indexed by rank, and then returns:
      - summary_df: DataFrame with one row per query, containing
          doc, label, label_count_top5/10/20, matched_rank, matched_score
      - results_map: dict mapping each doc.doc → its full ranked DataFrame
    """
    # 1) ONE‑TIME LOAD of heavy resources
    resources = aux_vsm.load_word2vec_resources(paths, use_multivector=use_multivector)

    results_map: Dict[str, pd.DataFrame] = {}
    summary = []

    for doc in documents:
        q = doc.query
        logging.info(f"Testing query: {q!r} at top_k={top_k}")
        out = aux_vsm.run_word2vec_query_preloaded(
            resources,
            query=q,
            top_k=top_k,
            use_expansion=use_expansion
        )

        # 2) Build the per-query DataFrame
        raw = out["results"]
        df = (
            pd.DataFrame(raw)
              .loc[:, ['rank', 'doc_id', 'score', 'grandparent']]
              .set_index('rank')
              .rename(columns={'grandparent': 'label'})
        )
        results_map[doc.doc] = df

        # 3) Compute summary metrics
        count5  = df.head(5) ['label'].eq(doc.label).sum()
        count10 = df.head(10)['label'].eq(doc.label).sum()
        count20 = df.head(20)['label'].eq(doc.label).sum()

        match = df[df['doc_id'] == doc.doc]
        if not match.empty:
            matched_rank  = int(match.index[0])
            matched_score = float(match['score'].iloc[0])
        else:
            matched_rank  = None
            matched_score = None

        summary.append({
            'doc':                doc.doc,
            'label':              doc.label,
            'label_count_top5':   count5,
            'label_count_top10':  count10,
            'label_count_top20':  count20,
            'matched_rank':       matched_rank,
            'matched_score':      matched_score
        })

    # 4) Build summary DataFrame
    summary_df = pd.DataFrame(summary)
    return summary_df, results_map

def evaluate_queries(
    paths: Dict[str, Path],
    documents: List["DocumentSection"],
    top_k: int = 20,
    method: str = "vsm",  # "vsm" or "bm25"
    resources: Optional[Union[dict, object]] = None,
    use_expansion: bool = True,
    use_multivector: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Evaluate VSM or BM25 queries on a list of documents.

    Returns:
        - summary_df: Evaluation summary as a DataFrame.
        - results_map: Mapping from doc ID to result DataFrame.
    """
    logger = logging.getLogger(__name__)
    summary = []
    results_map: Dict[str, pd.DataFrame] = {}

    # --- setup query_runner depending on method ---
    if method == "vsm":
        if resources is None:
            logger.info("Loading Word2Vec resources...")
            resources = aux_vsm.load_word2vec_resources(
                paths,
                use_multivector=use_multivector
            )
        query_runner = lambda q: aux_vsm.run_word2vec_query_preloaded(
            resources=resources,
            query=q,
            top_k=top_k,
            use_expansion=use_expansion
        )

    elif method == "bm25":
        if resources is None:
            retriever_path = paths["retriever"]
            if not retriever_path.exists():
                logger.info("Creating BM25 index...")
                resources = aux_bm25.create_bm25_index(
                    paths["pdf_folder"],
                    retriever_path
                )
            else:
                logger.info("Loading BM25 model from %s", retriever_path)
                resources = bm25s.BM25.load(
                    str(retriever_path),
                    load_corpus=True
                )
        corpus = getattr(resources, "corpus", None)

        # Now query_bm25 returns a List[BM25Result]
        query_runner = lambda q: aux_bm25.query_bm25(
            retriever_path=paths["retriever"],
            retriever=resources,
            query=q,
            k=top_k,
            corpus=corpus
        )

    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'vsm' or 'bm25'.")

    # --- run through all documents ---
    for doc in documents:
        try:
            logger.info(f"Running query for doc: {doc.doc} ({doc.label})")
            raw_result = query_runner(doc.query)

            # If BM25, unpack dataclass instances to dicts
            if method == "bm25" and raw_result and isinstance(raw_result[0], aux_bm25.BM25Result):
                records = [r.to_dict() for r in raw_result]
                df_raw = pd.DataFrame(records)
            else:
                # VSM: result is dict with "results" key
                df_raw = pd.DataFrame(raw_result["results"] if method == "vsm" else raw_result)

            # Select & rename columns dynamically
            if method == "vsm":
                df = (
                    df_raw.loc[:, ["rank", "doc_id", "score", "grandparent"]]
                          .rename(columns={"grandparent": "label"})
                )
            else:  # bm25
                df = (
                    df_raw.loc[:, ["rank", "doc_id", "doc_name", "label", "score"]]
                )

            df = df.set_index("rank")
            results_map[doc.doc] = df

            # Compute summary metrics
            counts = {
                "label_count_top5":  df.head(5)["label"].eq(doc.label).sum(),
                "label_count_top10": df.head(10)["label"].eq(doc.label).sum(),
                "label_count_top20": df.head(20)["label"].eq(doc.label).sum()
            }
            if method == "vsm":
                match = df[df["doc_id"] == doc.doc]
            else:  # for BM25, match on doc_name
                match = df[df["doc_name"] == doc.doc]

            matched_rank  = int(match.index[0])         if not match.empty else None
            matched_score = float(match["score"].iloc[0]) if not match.empty else None

            summary.append({
                "doc": doc.doc,
                "label": doc.label,
                **counts,
                "matched_rank": matched_rank,
                "matched_score": matched_score
            })

        except Exception as e:
            logger.error(f"Error processing {doc.doc}: {e}")
            summary.append({
                "doc": doc.doc,
                "label": doc.label,
                "label_count_top5": None,
                "label_count_top10": None,
                "label_count_top20": None,
                "matched_rank": None,
                "matched_score": None
            })

    summary_df = pd.DataFrame(summary)
    return summary_df, results_map