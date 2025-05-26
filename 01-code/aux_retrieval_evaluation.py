import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_document_retrieval_hybrid as aux_hybrid

import aux_semantic_search as aux_semantics
import bm25s


@dataclass
class DocumentSection:
    doc: str
    label: str
    query: str

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


def evaluate_hybrid_queries(
    paths: Dict[str, Path],
    documents: List["DocumentSection"],
    top_k: int = 20,
    weight_vsm: float = 0.6,
    weight_bm25: float = 0.4,
    norm_vsm: aux_hybrid.ScoreNormMethod = 'minmax',
    norm_bm25: aux_hybrid.ScoreNormMethod = 'minmax',
    use_expansion: bool = True,
    use_multivector: bool = True
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    1) Load VSM & BM25 resources
    2) Prepare normalization
    3) For each document: run VSM + BM25, normalize, fuse
    4) Package hybrid results into DataFrames + summary
    """
    logger = logging.getLogger(__name__)
    summary = []
    results_map: Dict[str, pd.DataFrame] = {}

    # === Step 1: Load resources once ===
    logger.info("Loading VSM resources...")
    vsm_resources = aux_vsm.load_word2vec_resources(
        paths, use_multivector=use_multivector
    )

    logger.info("Loading / building BM25 resources...")
    retriever_path = paths["retriever"]
    if not retriever_path.exists():
        resources_bm25 = aux_bm25.create_bm25_index(paths["pdf_folder"], retriever_path)
    else:
        resources_bm25 = bm25s.BM25.load(str(retriever_path), load_corpus=True)
    corpus = getattr(resources_bm25, "corpus", None)

    # === Step 2: nothing more to prepare beyond norm params ===

    # === Step 3: per-document hybrid search ===
    for doc in documents:
        try:
            logger.info(f"Hybrid querying for doc: {doc.doc} ({doc.label})")

            # 3a) VSM raw results (list of dicts)
            vsm_raw = aux_vsm.run_word2vec_query_preloaded(
                resources=vsm_resources,
                query=doc.query,
                top_k=top_k,
                use_expansion=use_expansion
            )["results"]

            # 3b) BM25 raw results (list of BM25Result)
            bm25_raw = aux_bm25.query_bm25(
                retriever_path=retriever_path,
                retriever=resources_bm25,
                query=doc.query,
                k=top_k,
                corpus=corpus
            )

            # 3c) Normalize scores
            #   VSM
            if norm_vsm in ('minmax','zscore'):
                aux_hybrid.normalize_scores(vsm_raw, 'score', 'vsm_norm', method=norm_vsm)
            else:
                for r in vsm_raw: r['vsm_norm'] = r.get('score', 0.0)
            #   BM25 (unpack dataclass → dict first)
            bm25_dicts = [
                (r.to_dict() if isinstance(r, aux_bm25.BM25Result) else r)
                for r in bm25_raw
            ]
            if norm_bm25 in ('minmax','zscore'):
                aux_hybrid.normalize_scores(bm25_dicts, 'score', 'bm25_norm', method=norm_bm25)
            else:
                for r in bm25_dicts: r['bm25_norm'] = r.get('score', 0.0)

            # 3d) Fuse via weighted sum
            fused = []
            vsm_map  = {r['doc_id']: r for r in vsm_raw}
            bm25_map = {r['doc_id']: r for r in bm25_dicts}
            all_ids  = set(vsm_map) | set(bm25_map)
            for did in all_ids:
                v = vsm_map.get(did, {})
                b = bm25_map.get(did, {})
                v_norm = v.get('vsm_norm',0.0)
                b_norm = b.get('bm25_norm',0.0)

                fused.append({
                    'doc_id': did,
                    'vsm_score':      v.get('score'),
                    'vsm_parent':     v.get('parent'),
                    'vsm_grandparent':v.get('grandparent'),
                    'vsm_norm':       v_norm,

                    'bm25_score':      b.get('score'),
                    'bm25_parent':     b.get('parent'),
                    'bm25_grandparent':b.get('grandparent'),
                    'bm25_norm':       b_norm,

                    'combined_score': v_norm*weight_vsm + b_norm*weight_bm25
                })

            # === Step 4: package into DataFrame & summary ===
            # assign ranks & choose a single 'label' column
            for idx, ent in enumerate(sorted(fused,
                                            key=lambda x: x['combined_score'],
                                            reverse=True), start=1):
                ent['rank'] = idx
                ent['label'] = ent.get('bm25_grandparent') or ent.get('vsm_grandparent')

            df = pd.DataFrame(fused).set_index('rank')
            results_map[doc.doc] = df

            # build summary metrics
            counts = {
                'label_count_top5':  df.head(5)['label'].eq(doc.label).sum(),
                'label_count_top10': df.head(10)['label'].eq(doc.label).sum(),
                'label_count_top20': df.head(20)['label'].eq(doc.label).sum()
            }
            match = df[df['doc_id']==doc.doc]
            mrank = int(match.index[0]) if not match.empty else None
            mscore= float(match['combined_score'].iloc[0]) if not match.empty else None

            summary.append({
                'doc': doc.doc,
                'label': doc.label,
                **counts,
                'matched_rank':  mrank,
                'matched_score': mscore
            })

        except Exception as e:
            logger.error(f"Error on {doc.doc}: {e}", exc_info=True)
            summary.append({
                'doc': doc.doc,
                'label': doc.label,
                'label_count_top5':  None,
                'label_count_top10': None,
                'label_count_top20': None,
                'matched_rank':      None,
                'matched_score':     None
            })

    summary_df = pd.DataFrame(summary)
    return summary_df, results_map

def optimize_hybrid_weights(
    aux_retrieval: Any,
    paths: Dict[str, Any],
    documents: List[Any],
    top_k: int = 200,
    norm_vsm: str = 'minmax',
    norm_bm25: str = 'minmax',
    use_expansion: bool = True,
    use_multivector: bool = False,
    weight_grid: Optional[List[Tuple[float, float]]] = None
) -> pd.DataFrame:
    """
    Iteratively evaluates hybrid retrieval over (weight_vsm, weight_bm25) pairs
    and returns a DataFrame with performance metrics:
      - median matched_rank
      - count of matched_rank < 10

    Args:
        aux_retrieval: Module or object with method evaluate_hybrid_queries
        paths: Retrieval paths configuration
        documents: List of DocumentSection objects
        top_k: Number of top results to retrieve per query
        norm_vsm: Normalization method for VSM ('minmax', 'zscore', or None)
        norm_bm25: Normalization method for BM25 ('minmax', 'zscore', or None)
        use_expansion: Whether to apply query expansion in VSM
        use_multivector: Whether to use multivector embeddings in VSM
        weight_grid: List of (weight_vsm, weight_bm25) pairs to test
                      Defaults to [(0.0,1.0), (0.1,0.9), ..., (1.0,0.0)]

    Returns:
        DataFrame sorted by lowest median_rank and highest count_lt10, columns:
        ['weight_vsm', 'weight_bm25', 'median_rank', 'count_lt10']
    """
    # Default weight grid: 0.0 to 1.0 in 0.1 steps
    if weight_grid is None:
        weight_grid = [(i / 10.0, 1.0 - i / 10.0) for i in range(11)]

    records = []
    for w_vsm, w_bm25 in weight_grid:
        # Evaluate hybrid retrieval
        summary_df, _ = aux_retrieval.evaluate_hybrid_queries(
            paths=paths,
            documents=documents,
            top_k=top_k,
            weight_vsm=w_vsm,
            weight_bm25=w_bm25,
            norm_vsm=norm_vsm,
            norm_bm25=norm_bm25,
            use_expansion=use_expansion,
            use_multivector=use_multivector
        )

        # Extract matched_rank and compute metrics
        if 'matched_rank' in summary_df.columns:
            ranks = summary_df['matched_rank'].dropna().astype(int)
        else:
            ranks = pd.Series([], dtype=int)

        median_rank = float(ranks.median()) if not ranks.empty else float('nan')
        count_lt10 = int((ranks < 10).sum())

        records.append({
            'weight_vsm': w_vsm,
            'weight_bm25': w_bm25,
            'median_rank': median_rank,
            'count_lt10': count_lt10
        })

    # Build and sort result DataFrame
    result_df = pd.DataFrame(records)
    result_df = result_df.sort_values(
        by=['median_rank', 'count_lt10'],
        ascending=[True, False]
    ).reset_index(drop=True)

    return result_df


##-----------------------------------------
##new refactor of absolutelly everything
import statistics

def compute_query_run_stats(
    records: List[Dict[str, Any]],
    top_k_values: List[int] = [5, 10, 20]
) -> Dict[str, pd.DataFrame]:
    """
    Calcula estadísticas de múltiples ejecuciones de consulta y devuelve DataFrames:

    - per_record: DataFrame con columnas ['doc','label','rank','score','label_count_top_{K}',...]
    - top_counts: DataFrame de conteos de cuántas veces la etiqueta original aparece en top-K
    - score_stats: DataFrame con max, min y median de scores de coincidencias exactas de doc
    """
    per_record = []
    label_counts = {k: 0 for k in top_k_values}
    all_scores = []

    for rec in records:
        doc_name = rec['doc']
        label = rec.get('label')
        retrieved: List[aux_vsm.RetrievedDocument] = rec['result'].results.documents

        rank = next((e.rank for e in retrieved if e.doc_id == doc_name), None)
        score = next((e.score for e in retrieved if e.doc_id == doc_name), None)
        if score is not None:
            all_scores.append(score)

        retrieved_sorted = sorted(retrieved, key=lambda e: e.rank)

        row = {'doc': doc_name, 'label': label, 'rank': rank, 'score': score}
        for k in top_k_values:
            top_k_entries = [e for e in retrieved_sorted if e.rank <= k]
            count_label = sum(1 for e in top_k_entries if e.label == label)
            row[f'label_count_top_{k}'] = count_label
            label_counts[k] += count_label

        per_record.append(row)

    df_per_record = pd.DataFrame(per_record)

    df_top_counts = pd.DataFrame(
        {'top_k': list(label_counts.keys()), 'label_occurrences': list(label_counts.values())}
    )

    if all_scores:
        stats = {'max': max(all_scores), 'min': min(all_scores), 'median': statistics.median(all_scores)}
    else:
        stats = {'max': None, 'min': None, 'median': None}
    df_score_stats = pd.DataFrame([stats])

    return {'per_record': df_per_record, 'top_counts': df_top_counts, 'score_stats': df_score_stats}
