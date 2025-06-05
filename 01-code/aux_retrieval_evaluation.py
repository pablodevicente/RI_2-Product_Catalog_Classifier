import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_document_retrieval_hybrid as aux_hybrid
import dataclass as data
import aux_semantic_search as aux_semantics
import bm25s
from dataclass import QueryResult, TopKDocumentsResult, RetrievedDocument
import matplotlib.pyplot as plt
import pickle
import seaborn as sns


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
                resources = aux_bm25.create_bm25_index(paths["pdf_folder"],retriever_path)
            else:
                logger.info("Loading BM25 model from %s", retriever_path)
                resources = bm25s.BM25.load(str(retriever_path),load_corpus=True)

        corpus = getattr(resources, "corpus", None)

        # Now query_bm25 returns a List[BM25Result]
        query_runner = lambda q: aux_bm25.query_bm25(
            retriever_path=retriever_path,
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
                    df_raw.loc[:, ["rank", "doc_id", "label", "score"]]
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
                match = df[df["doc_id"] == doc.doc]

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

def rrf_from_dfs(
    dfs: List[pd.DataFrame],
    rrf_k: int = 60,
    top_k: int = 10,
    multi_vector: bool = True
) -> data.QueryResult:
    """
    Reciprocal Rank Fusion over multiple pandas DataFrames.

    :param dfs: list of DataFrames, each with columns
                ['rank', 'score', 'label', 'doc_id']
    :param rrf_k: the k parameter in RRF (defaults to 60)
    :param top_k: how many fused docs to return
    :param multi_vector: flag to store in TopKDocumentsResult
    :returns: RRFQueryResult(results=TopKDocumentsResult(...))
    """
    # for each DataFrame, compute the 1/(rrf_k + rank) contribution
    contribs = []
    for df in dfs:
        tmp = pd.DataFrame({
            "doc_id": df["doc_id"],
            "rrf": 1.0 / (rrf_k + df["rank"]),
            "label": df["label"],
        })
        contribs.append(tmp)

    # stack and sum per doc_id (and grab a label)
    all_contrib = pd.concat(contribs, ignore_index=True)
    summed = (
        all_contrib
        .groupby(["doc_id", "label"], as_index=False)
        .agg(rrf_score=("rrf", "sum"))
    )

    # pick top_k by that fused score
    top = summed.nlargest(top_k, "rrf_score").reset_index(drop=True)

    # build RetrievedDocument list
    fused_docs = [
        data.RetrievedDocument(
            rank=i+1,
            doc_id=row.doc_id,
            score=row.rrf_score,
            label=row.label
        )
        for i, row in top.iterrows()
    ]

    # wrap into TopKDocumentsResult and RRFQueryResult
    topk_res = data.TopKDocumentsResult(
        top_k=top_k,
        documents=fused_docs,
        multi_vector=multi_vector
    )
    return data.QueryResult(results=topk_res)

def hybrid_search(
    bm25_results: List[Dict[str, Any]],
    vsm_results: List[Dict[str, Any]],
    weight_bm25: float,
    weight_vsm:  float,
    top_k:       int                   = 10,
    norm_bm25:   Optional[aux_hybrid.ScoreNormMethod] = None,
    norm_vsm:    Optional[aux_hybrid.ScoreNormMethod] = None,
    multi_vector: bool                 = True
) -> QueryResult:
    """
    Weighted hybrid search over BM25 + VSM.

    :param bm25_results: list of dicts with at least ['doc_id','score','label']
    :param vsm_results:  same structure as bm25_results
    :param weight_bm25:  weight to apply to BM25 scores
    :param weight_vsm:   weight to apply to VSM scores
    :param top_k:        how many final docs to return
    :param norm_bm25:    'minmax' or 'zscore' to normalize BM25, else raw
    :param norm_vsm:     'minmax' or 'zscore' to normalize VSM, else raw
    :param multi_vector: flag stored in TopKDocumentsResult
    :returns:            TopKDocumentsResult with fused RetrievedDocument list
    """
    # 1) Normalize or copy raw
    if norm_bm25 in ('minmax', 'zscore'):
        aux_hybrid.normalize_scores(bm25_results, score_key='score', norm_key='bm25_norm', method=norm_bm25)
    else:
        for r in bm25_results:
            r['bm25_norm'] = r.get('score', 0.0)

    if norm_vsm in ('minmax', 'zscore'):
        aux_hybrid.normalize_scores(vsm_results, score_key='score', norm_key='vsm_norm', method=norm_vsm)
    else:
        for r in vsm_results:
            r['vsm_norm'] = r.get('score', 0.0)

    # 2) Merge by doc_id, summing up if duplicates
    merged: Dict[str, Dict[str, Any]] = {}
    for r in bm25_results + vsm_results:
        doc = merged.setdefault(r['doc_id'], {
            'doc_id':   r['doc_id'],
            'label':    r['label'],
            'bm25_norm': 0.0,
            'vsm_norm':  0.0,
        })
        # add whichever norm exists
        doc['bm25_norm'] += r.get('bm25_norm', 0.0)
        doc['vsm_norm']  += r.get('vsm_norm',  0.0)

    # 3) Compute weighted hybrid score
    for doc in merged.values():
        doc['hybrid_score'] = weight_bm25 * doc['bm25_norm'] + weight_vsm * doc['vsm_norm']

    # 4) Sort & take top_k
    top = sorted(
        merged.values(),
        key=lambda d: d['hybrid_score'],
        reverse=True
    )[:top_k]

    # 5) Build RetrievedDocument list
    fused_docs = [
        RetrievedDocument(
            rank= idx + 1,
            doc_id= d['doc_id'],
            score= d['hybrid_score'],
            label= d['label']
        )
        for idx, d in enumerate(top)
    ]

    # 6) Wrap into TopKDocumentsResult
    topk_hybrid = TopKDocumentsResult(
        top_k=       top_k,
        documents=   fused_docs,
        multi_vector=multi_vector
    )

    return QueryResult(
        results=topk_hybrid,
    )

def run_hybrid_query(
    paths: Dict[str, Path],
    query: str,
    top_k: int = 50,
    use_multivector : bool = True
) -> QueryResult:
    # 1. BM25
    bm25 = aux_bm25.run_bm25_query(paths, query, top_k=top_k)

    # 2. VSM
    resources = aux_vsm.load_word2vec_resources(paths,use_multivector=use_multivector)
    vsm = aux_vsm.run_word2vec_query_preloaded(resources, query, top_k=top_k*2, use_expansion=False)
    vsm_lookup = { d.doc_id: d.score for d in vsm.results.documents }

    # 3. Build and 4. Rerank
    drafts = []
    for base in bm25.results.documents:
        drafts.append(RetrievedDocument(
            rank=0,
            doc_id=base.doc_id,
            label=base.label,
            score=vsm_lookup.get(base.doc_id, 0.0)
        ))
    reranked = sorted(drafts, key=lambda d: d.score, reverse=True)
    for i, d in enumerate(reranked, 1):
        d.rank = i

    # 5. Wrap
    hybrid_topk = TopKDocumentsResult(top_k=top_k, documents=reranked, multi_vector=True)
    return QueryResult(results=hybrid_topk)


# 3) DEFINE A FUNCTION TO PLOT Recall@5, @10, @20 FOR ALL DOCUMENTS
# -------------------------------------------------------------------
def plot_recall_for_documents(df, system_name):
    """
    For a single pandas DataFrame `df` (with columns
    ['label_count_top_5','label_count_top_10','label_count_top_20']),
    this function plots three lines:
      - label_count_top_5   (Recall @ 5)
      - label_count_top_10  (Recall @ 10)
      - label_count_top_20  (Recall @ 20)

    Horizontal axis: document index (1, 2, …, len(df))
    Vertical axis: number of labels (range 0–20)
    """
    num_docs = len(df)
    x = range(1, num_docs + 1)  # 1-based document indices

    plt.figure(figsize=(10, 6))
    plt.plot(x, df['label_count_top_5'],  marker='o', label='Recall @ 5')
    plt.plot(x, df['label_count_top_10'], marker='o', label='Recall @ 10')
    plt.plot(x, df['label_count_top_20'], marker='o', label='Recall @ 20')

    plt.xlabel('Document Index')
    plt.ylabel('Recall (Count of Relevant Labels)')
    plt.title(f'Recall@5, @10, @20 for {system_name} (All Documents)')
    plt.ylim(0, 20)
    plt.xticks(x)            # Show every document on the x-axis
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


# 3) DEFINE THE FUNCTION TO PLOT A GIVEN "recall" COLUMN FOR ALL METHODS
def plot_combined_recall(df_list, recall_col):
    """
    Plots the specified recall column (e.g., 'label_count_top_5')
    for all ranking systems in df_list on a single chart.

    Parameters:
    - df_list: List of tuples [(name, DataFrame), ...]
    - recall_col: str, one of ['label_count_top_5', 'label_count_top_10', 'label_count_top_20']
    """
    # Assume all DataFrames have the same number of rows (documents)
    num_docs = len(df_list[0][1])
    x = range(1, num_docs + 1)   # Document indices are 1-based

    plt.figure(figsize=(10, 6))
    for name, df in df_list:
        plt.plot(
            x,
            df[recall_col],
            marker='o',
            label=name
        )

    plt.xlabel('Document Index')
    plt.ylabel(f'Recall from column: {recall_col}')
    plt.title(f'Combined Comparison of {recall_col} Across Ranking Systems')
    # Set y-axis limit so it’s just above the max value found (for visual clarity)
    plt.ylim(0, df_list[0][1][recall_col].max() + 1)
    plt.xticks(x)  # Show every document index on the x-axis
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_combined_rank(df_list, cap=50):
    """
    Plots the 'rank' column for all ranking systems in df_list on a single chart,
    capping any rank > cap (and NaN) to the value of cap before plotting.

    Parameters:
    - df_list: list of tuples (name: str, DataFrame) where each DataFrame contains a 'rank' column.
    - cap: int (default=50). Any rank value greater than cap, or NaN, will be replaced by cap.

    This draws one line per ranking system:
      • x-axis = document index (1, 2, …, N)
      • y-axis = capped rank (position) for each document in that system
    """
    # Assume all DataFrames have the same number of rows (documents)
    num_docs = len(df_list[0][1])
    x = range(1, num_docs + 1)  # Document indices (1-based)

    plt.figure(figsize=(10, 6))
    for name, df in df_list:
        # Ensure we don’t modify the original DataFrame:
        rank_series = df['rank'].copy()

        # Replace NaN with cap, then cap any value > cap
        rank_capped = rank_series.fillna(cap).clip(upper=cap)

        plt.plot(
            x,
            rank_capped,
            marker='o',
            label=name
        )

    plt.xlabel('Document Index')
    plt.ylabel(f'Rank (capped at {cap})')
    plt.title(f'Combined Comparison of Rank Across Ranking Systems (cap={cap})')
    plt.xticks(x)  # Show every document index on the x-axis
    plt.ylim(0, cap + 1)  # y-axis from 0 to cap+1 for clarity
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def load_rqe_hybrid_runs(start, end):
    """
    Iteratively loads pickle files named
    '../02-data/06-Evaluation/ranking_query_evaluation_{i}.pkl' for i in [start, end],
    and extracts the 'hybrid_statistics' DataFrame from each.

    Parameters:
    - start: int, starting index
    - end:   int, ending index (inclusive)

    Returns:
    - List of pandas.DataFrame objects, each corresponding to the
      'hybrid_statistics' DataFrame from ranking_query_evaluation_i.pkl.
    """
    hybrid_runs = []
    for i in range(start, end + 1):

        filename = f'../02-data/06-Evaluation/ranking_query_evaluation_{i}.pkl'
        with open(filename, 'rb') as f:
            rqe_dict = pickle.load(f)
            # Extract the 'hybrid_statistics' DataFrame
            hybrid_df = rqe_dict['hybrid_statistics']
            hybrid_runs.append(hybrid_df)
    return hybrid_runs


def plot_hybrid_across_evaluations(hybrid_items, metric_col, labels=None, cap=50):
    """
    Plots a chosen metric column from multiple Hybrid results, where each element of `hybrid_items`
    is either:
      - a pandas.DataFrame already containing the Hybrid results, or
      - a dict with key 'hybrid_statistics' whose value is the DataFrame we want.

    Parameters:
    ----------
    hybrid_items : list of (pandas.DataFrame or dict)
        Each item is either:
          • A DataFrame corresponding to Hybrid results for one evaluation run, or
          • A dict that has key 'hybrid_statistics' (the DataFrame of interest).
        All Hybrid-DataFrames must have the same number of rows (same set of documents).

    metric_col : str
        The column name to plot:
        e.g. 'label_count_top_5', 'label_count_top_10', 'label_count_top_20', or 'rank'.

    labels : list of str or None
        If provided, a list of the same length as `hybrid_items`, giving each curve a legend label
        (e.g. ['Run 1', 'Run 2', …, 'Run 6']). If None, the function auto-generates: 'Eval_1', 'Eval_2', …

    cap : int or None (default=50)
        If not None, then any value in `metric_col` that is > cap or NaN will be replaced by cap before plotting.
        For example, cap=50 forces all values >50 (and NaN) down to 50.

    Behavior:
    --------
    - Checks each element in `hybrid_items`:
        • If it’s a dict containing 'hybrid_statistics', it uses that DataFrame.
        • Otherwise, it treats the element itself as the DataFrame.
    - Expects all resulting DataFrames have the same # of rows.
    - Draws one line per evaluation, plotting doc index (1…N) on x-axis and `metric_col` (capped) on y-axis.
    """
    # First, extract the DataFrame from each element
    dfs = []
    for idx, item in enumerate(hybrid_items):
        if isinstance(item, dict):
            if 'hybrid_statistics' not in item:
                raise KeyError(f"Item {idx} is a dict but has no 'hybrid_statistics' key.")
            dfs.append(item['hybrid_statistics'])
        else:
            # Assume item itself is a DataFrame
            dfs.append(item)

    # Check they all have the same number of rows
    num_docs = len(dfs[0])
    for idx, df in enumerate(dfs):
        if len(df) != num_docs:
            raise ValueError(f"DataFrame at index {idx} has {len(df)} rows, but expected {num_docs} rows.")

    x = range(1, num_docs + 1)  # 1-based document indices

    # Auto-generate labels if not provided
    if labels is None:
        labels = [f'Eval_{i + 1}' for i in range(len(dfs))]
    if len(labels) != len(dfs):
        raise ValueError("Length of 'labels' must match number of runs (len(hybrid_items)).")

    plt.figure(figsize=(10, 6))

    for df, lbl in zip(dfs, labels):
        # Copy to avoid modifying original
        series = df[metric_col].copy()

        # If cap is specified, replace NaNs with cap and clip above cap
        if cap is not None:
            series = series.fillna(cap).clip(upper=cap)

        plt.plot(
            x,
            series,
            marker='o',
            label=lbl
        )

    plt.xlabel('Document Index')
    ylabel = metric_col
    if cap is not None:
        ylabel += f' (capped at {cap})'
    plt.ylabel(ylabel)

    plt.title(f'Hybrid: Comparison of "{metric_col}" Across {len(dfs)} Evaluations')
    plt.xticks(x)  # Show every document index on the x-axis

    # If plotting 'rank' and we've capped it, force the y-limit
    if cap is not None and metric_col.lower().strip() == 'rank':
        plt.ylim(0, cap + 1)

    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def plot_hybrid_across_evaluations_sns(hybrid_items, metric_col, labels=None, cap=50):
    """
    Plots a chosen metric column from multiple Hybrid results using seaborn, where each element of `hybrid_items`
    is either:
      - a pandas.DataFrame already containing the Hybrid results, or
      - a dict with key 'hybrid_statistics' whose value is the DataFrame we want.

    Parameters:
    ----------
    hybrid_items : list of (pandas.DataFrame or dict)
        Each item is either:
          • A DataFrame corresponding to Hybrid results for one evaluation run, or
          • A dict that has key 'hybrid_statistics' (the DataFrame of interest).
        All Hybrid-DataFrames must have the same number of rows (same set of documents).

    metric_col : str
        The column name to plot:
        e.g. 'label_count_top_5', 'label_count_top_10', 'label_count_top_20', or 'rank'.

    labels : list of str or None
        If provided, a list of the same length as `hybrid_items`, giving each curve a legend label
        (e.g. ['Run 1', 'Run 2', …, 'Run 6']). If None, the function auto-generates: 'Eval_1', 'Eval_2', …

    cap : int or None (default=50)
        If not None, then any value in `metric_col` that is > cap or NaN will be replaced by cap before plotting.
        For example, cap=50 forces all values >50 (and NaN) down to 50.

    Behavior:
    --------
    - Checks each element in `hybrid_items`:
        • If it’s a dict containing 'hybrid_statistics', it uses that DataFrame.
        • Otherwise, it treats the element itself as the DataFrame.
    - Expects all resulting DataFrames have the same # of rows.
    - Draws one seaborn lineplot per evaluation, plotting doc index (1…N) on x-axis and `metric_col` (capped) on y-axis.
    """

    # 1) Extract the DataFrame from each element
    dfs = []
    for idx, item in enumerate(hybrid_items):
        if isinstance(item, dict):
            if 'hybrid_statistics' not in item:
                raise KeyError(f"Item {idx} is a dict but has no 'hybrid_statistics' key.")
            dfs.append(item['hybrid_statistics'])
        else:
            # Assume item itself is a DataFrame
            dfs.append(item)

    # 2) Verify all DataFrames have the same number of rows
    num_docs = len(dfs[0])
    for idx, df in enumerate(dfs):
        if len(df) != num_docs:
            raise ValueError(f"DataFrame at index {idx} has {len(df)} rows, but expected {num_docs} rows.")

    x = range(1, num_docs + 1)  # 1-based document indices

    # 3) Auto-generate labels if not provided
    if labels is None:
        labels = [f'Eval_{i + 1}' for i in range(len(dfs))]
    if len(labels) != len(dfs):
        raise ValueError("Length of 'labels' must match number of runs (len(hybrid_items)).")

    # 4) Set up seaborn theme and palette
    sns.set_theme(style="whitegrid", palette="Set2")

    fig, ax = plt.subplots(figsize=(10, 6))

    # 5) Plot each series with seaborn
    for df, lbl in zip(dfs, labels):
        # Copy to avoid modifying the original
        series = df[metric_col].copy()

        # If cap is specified, replace NaNs with cap and clip above cap
        if cap is not None:
            series = series.fillna(cap).clip(upper=cap)

        # Use seaborn lineplot for each run
        sns.lineplot(
            x=list(x),
            y=series,
            marker="o",
            label=lbl,
            ax=ax
        )

    # 6) Labeling and formatting
    ax.set_xlabel("Document Index")
    ylabel = metric_col
    if cap is not None:
        ylabel += f" (capped at {cap})"
    ax.set_ylabel(ylabel)
    ax.set_title(f'Hybrid: Comparison of "{metric_col}" Across {len(dfs)} Evaluations')

    ax.set_xticks(list(x))  # Show every document index on the x-axis

    # If plotting 'rank' and we've capped it, force the y-limit
    if cap is not None and metric_col.lower().strip() == 'rank':
        ax.set_ylim(0, cap + 1)

    ax.legend()
    plt.tight_layout()
    plt.show()


def run_all_models(query_list_documents: List[Any], paths: Dict[str, str], use_expansion: bool = True,
                   use_multivector: bool = True, top_k : int = 50, norm_vsm: str = 'minmax', norm_bm25:str = 'zscore') -> Dict[str, Any]:
    """
    Function to run all retrieval models (BM25, RRF, Word2Vec, Hybrid) and compute statistics.

    Args:
    - query_list_documents: List of query documents containing query, doc, and label.
    - paths: Dictionary of paths required for the models.
    - use_expansion: Boolean flag for Word2Vec query expansion.
    - use_multivector: Boolean flag for using multivector resources.

    Returns:
    - stats_data: Dictionary containing statistics data for each retrieval model.
    """


    vsm_records = []
    bm25_records = []
    rrf_records = []
    hybrid_records = []
    rerank_records = []

    # Load resources for Word2Vec
    resources = aux_vsm.load_word2vec_resources(paths, use_multivector=use_multivector)

    logger = logging.getLogger(__name__)

    # Run BM25 for each query
    for section in query_list_documents:
        bm25_result = aux_bm25.run_bm25_query(paths, section.query, top_k=top_k)
        bm25_records.append({
            "doc": section.doc,
            "label": section.label,
            "query": section.query,
            "result": bm25_result
        })

    # Compute BM25 statistics
    bm25_statistics = compute_query_run_stats(bm25_records)

    # Run Word2Vec (VSM) for each query
    for section in query_list_documents:
        w2v_result = aux_vsm.run_word2vec_query_preloaded(resources, section.query, top_k=top_k,
                                                          use_expansion=use_expansion)
        vsm_records.append({
            "doc": section.doc,
            "label": section.label,
            "query": section.query,
            "result": w2v_result
        })

    # Compute VSM statistics
    vsm_statistics = compute_query_run_stats(vsm_records)

    # Run RRF using BM25 and VSM results
    for i, section in enumerate(query_list_documents):
        df_bm25 = pd.DataFrame([d.__dict__ for d in bm25_records[i]["result"].results.documents])
        df_vsm = pd.DataFrame([d.__dict__ for d in vsm_records[i]["result"].results.documents])
        rrf_result = rrf_from_dfs([df_bm25, df_vsm], rrf_k=60, top_k=top_k)
        rrf_records.append({
            "doc": section.doc,
            "label": section.label,
            "query": section.query,
            "result": rrf_result
        })

    # Compute RRF statistics
    rrf_statistics = compute_query_run_stats(rrf_records)

    # Run Hybrid model (assuming it combines results from BM25 and Word2Vec)
    for section in query_list_documents:
        hybrid_result = hybrid_search(paths=paths, query=section.query, top_k=top_k,
                                                       use_multivector=use_multivector,norm_bm25=norm_bm25,norm_vsm=norm_vsm)
        hybrid_records.append({
            "doc": section.doc,
            "label": section.label,
            "query": section.query,
            "result": hybrid_result
        })

    # Compute Hybrid statistics
    hybrid_statistics = compute_query_run_stats(hybrid_records)

    # Run reranking model (assuming it reranks results from the Hybrid model)
    top_k_rerank = 50  # Adjust top_k for reranking if needed
    for section in query_list_documents:
        rerank_results = run_hybrid_query(paths=paths, query=section.query, top_k=top_k_rerank,
                                                        use_multivector=use_multivector)
        rerank_records.append({
            "doc": section.doc,
            "label": section.label,
            "query": section.query,
            "result": rerank_results
        })

    # Compute rerank statistics
    rerank_statistics = compute_query_run_stats(rerank_records)

    # Construct stats_data dictionary
    stats_data = {
        'bm25_statistics': bm25_statistics['per_record'],
        'vsm_statistics': vsm_statistics['per_record'],
        'rrf_statistics': rrf_statistics['per_record'],
        'hybrid_statistics': hybrid_statistics['per_record'],
        'rerank_statistics': rerank_statistics['per_record']
    }

    return stats_data
