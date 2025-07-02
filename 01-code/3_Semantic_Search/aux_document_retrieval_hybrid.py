from typing import List, Dict, Any, Literal, Optional
import logging
import aux_document_retrieval_vsm as aux_vsm
import aux_document_retrieval_bm25 as aux_bm25
import pandas as pd

# Module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

ScoreNormMethod = Literal['minmax', 'zscore', None]

def normalize_scores(
    results: List[Dict[str, Any]],
    score_key: str = 'score',
    norm_key: str = None,
    method: ScoreNormMethod = 'minmax'
) -> List[Dict[str, Any]]:
    """
    In-place normalize the `score_key` values in `results` using:
      - 'minmax': (x - min) / (max - min)  → [0,1]
      - 'zscore': (x - mean) / std         → ℝ

    Args:
        results:   List of dicts, each with a numeric `score_key`.
        score_key: Which key to normalize (default 'score').
        norm_key:  Key under which to store the normalized value.
                   Defaults to f"{score_key}_{method}".
        method:    'minmax' or 'zscore'.

    Returns:
        The same `results` list, mutated in-place with each dict
        gaining `dict[norm_key] = normalized_value`.
    """
    if norm_key is None:
        norm_key = f"{score_key}_{method}"

    # Collect raw scores
    scores = [item.get(score_key, 0.0) for item in results]
    if not scores:
        logger.warning("No scores to normalize.")
        return results

    if method == 'minmax':
        min_s, max_s = min(scores), max(scores)
        span = max_s - min_s
        if span == 0:
            logger.info("All scores equal; skipping min-max normalization.")
            return results
        for item in results:
            item[norm_key] = (item.get(score_key, 0.0) - min_s) / span

    elif method == 'zscore':
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std = variance ** 0.5
        if std == 0:
            logger.info("Zero standard deviation; skipping z-score normalization.")
            return results
        for item in results:
            item[norm_key] = (item.get(score_key, 0.0) - mean) / std

    else:
        logger.error(f"Unknown normalization method: {method!r}")
        return results

    return results


def hybrid_retrieval(
        top_k_vsm: Dict[str, Any],
        top_k_bm25: List[Dict[str, Any]],
        weight_vsm: float = 0.6,
        weight_bm25: float = 0.4,
        norm_vsm: ScoreNormMethod = 'minmax',
        norm_bm25: ScoreNormMethod = 'minmax'
) -> List[Dict[str, Any]]:
    """
    Optionally normalize VSM and BM25 scores, then fuse via weighted sum.

    Args:
        top_k_vsm:  { "results": [ {doc_id, score, rank, parent, grandparent, ...}, ... ] }
        top_k_bm25: [ {doc_id, score, rank, parent, grandparent, ...}, ... ]
        weight_vsm: weight for vsm_norm in the combined score
        weight_bm25: weight for bm25_norm in the combined score
        norm_vsm:   'minmax', 'zscore', or None to skip VSM normalization
        norm_bm25:  'minmax', 'zscore', or None to skip BM25 normalization

    Returns:
        [ { doc_id, vsm_*, bm25_*, combined_score }, ... ] sorted desc.
    """
    vsm_results = top_k_vsm.get("results", [])
    bm25_results = top_k_bm25

    # 1) Normalize (or copy raw) into vsm_norm and bm25_norm
    if norm_vsm in ('minmax', 'zscore'):
        # writes into key 'vsm_norm'
        normalize_scores(
            vsm_results,
            score_key='score',
            norm_key='vsm_norm',
            method=norm_vsm
        )
    else:
        # no normalization → copy raw
        for r in vsm_results:
            r['vsm_norm'] = r.get('score', 0.0)

    if norm_bm25 in ('minmax', 'zscore'):
        normalize_scores(
            bm25_results,
            score_key='score',
            norm_key='bm25_norm',
            method=norm_bm25
        )
    else:
        for r in bm25_results:
            r['bm25_norm'] = r.get('score', 0.0)

    # 2) Build lookup maps by doc_id
    vsm_map = {r["doc_id"]: r for r in vsm_results}
    bm25_map = {r["doc_id"]: r for r in bm25_results}

    # 3) Union of all doc_ids
    all_docs = set(vsm_map) | set(bm25_map)

    # 4) Merge
    fused: List[Dict[str, Any]] = []
    for doc_id in all_docs:
        v = vsm_map.get(doc_id, {})
        b = bm25_map.get(doc_id, {})

        vsm_norm = v.get("vsm_norm", 0.0)
        bm25_norm = b.get("bm25_norm", 0.0)

        fused.append({
            "doc_id": doc_id,

            "vsm_rank": v.get("rank"),
            "vsm_score": v.get("score"),
            "vsm_parent": v.get("parent"),
            "vsm_grandparent": v.get("grandparent"),
            "vsm_norm": vsm_norm,

            "bm25_rank": b.get("rank"),
            "bm25_score": b.get("score"),
            "bm25_parent": b.get("parent"),
            "bm25_grandparent": b.get("grandparent"),
            "bm25_norm": bm25_norm,

            "combined_score": vsm_norm * weight_vsm + bm25_norm * weight_bm25
        })

    # 5) Sort descending by combined_score
    fused.sort(key=lambda x: x["combined_score"], reverse=True)
    return fused


def rrf(
    vsm_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: float = 60.0
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF):
      For each document:
        rrf_vsm   = 1.0 / (vsm_rank + k)
        rrf_bm25  = 1.0 / (bm25_rank + k)
        rrf_score = rrf_vsm + rrf_bm25

    Args:
        vsm_results:  List of dicts, each with keys "doc_id", "rank", plus metadata.
        bm25_results: List of dicts, each with keys "doc_id", "rank", plus metadata.
        k:            RRF constant (default 60).

    Returns:
        A merged list of dicts, one per doc_id, containing:
          - doc_id
          - all side-specific metadata (prefixed vsm_ or bm25_)
          - rrf_vsm, rrf_bm25, rrf_score
    """
    vsm_map  = {r["doc_id"]: r for r in vsm_results}
    bm25_map = {r["doc_id"]: r for r in bm25_results}

    all_ids = set(vsm_map) | set(bm25_map)
    fused: List[Dict[str, Any]] = []

    for doc_id in all_ids:
        v = vsm_map.get(doc_id, {})
        b = bm25_map.get(doc_id, {})

        v_rank = v.get("rank", len(vsm_results) + 1)
        b_rank = b.get("rank", len(bm25_results) + 1)

        rrf_v = 1.0 / (v_rank + k)
        rrf_b = 1.0 / (b_rank + k)
        total_rrf = rrf_v + rrf_b

        fused.append({
            "doc_id":      doc_id,

            **{f"vsm_{key}":  v.get(key) for key in ("rank","score","grandparent","parent")},

            **{f"bm25_{key}": b.get(key) for key in ("rank","score","grandparent","parent")},

            "rrf_vsm":     rrf_v,
            "rrf_bm25":    rrf_b,
            "rrf_score":   total_rrf,
        })

    # sort by final RRF score descending
    fused.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused


def print_documents(document_list: List[Dict[str, Any]], top_k: int, ranking: str) -> None:
    """
    Prints the top_k entries from document_list, auto‑detecting whether
    each entry uses RRF fusion, hybrid fusion (combined_score), or a plain
    BM25‑VSM rerank (bm25_score + vsm_score).
    """

    logger.info(f"--- Showing top {top_k} results for {ranking} ranking ---")

    for rank, entry in enumerate(document_list[:top_k], start=1):
        # Common metadata: try rerank fields first, then legacy keys
        gp = entry.get("grandparent") or entry.get("vsm_grandparent") or entry.get("bm25_grandparent") or "—"
        p  = entry.get("parent")      or entry.get("vsm_parent")       or entry.get("bm25_parent")       or "—"

        if "rrf_score" in entry:
            # RRF fusion
            logger.info(
                f"{rank}. {gp} : {p} "
                f"(RRF score: {entry['rrf_score']:.6f}, "
                f"VSM component: {entry['rrf_vsm']:.6f}, "
                f"BM25 component: {entry['rrf_bm25']:.6f})"
            )

        elif "combined_score" in entry:
            # Hybrid fusion
            logger.info(
                f"{rank}. {gp} : {p} "
                f"(combined: {entry['combined_score']:.4f}, "
                f"VSM: {entry['vsm_norm']:.4f}, "
                f"BM25: {entry['bm25_norm']:.4f})"
            )

        elif "bm25_score" in entry and "vsm_score" in entry:
            # Plain BM25-VSM rerank
            doc_id = entry.get("doc_id", "—")
            path   = entry.get("path") or "—"
            logger.info(
                f"{rank}. {gp} : {p} "
                f"(BM25: {entry['bm25_score']:.4f}, VSM: {entry['vsm_score']:.4f}) "
                f"[doc_id: {doc_id}, path: {path}]"
            )

        else:
            # Fallback: just dump the entry
            logger.info(f"{rank}. {gp} : {p} | data: {entry}")

def rerank(paths, query, top_k, mode="bm25-vsm"):

    # first bm25 and then rerank based on vsm
    if mode == "bm25-vsm":
        # 1) BM25 shortlist
        top_bm25 = aux_bm25.run_bm25_query(paths, query, top_k=top_k)
        bm25_ids = [doc["doc_id"] for doc in top_bm25]
        bm25_scores = {doc["doc_id"]: doc["score"] for doc in top_bm25}
        bm25_parents = {doc["doc_id"]: doc.get("parent") for doc in top_bm25}
        bm25_grandparents = {doc["doc_id"]: doc.get("grandparent") for doc in top_bm25}

        # 2) VSM scores for BM25 shortlist
        vsm_output = aux_vsm.run_word2vec_query_preloaded(paths, query, top_k=100, use_expansion=True)

        vsm_results = vsm_output.get("results", [])
        # Build full-map with a default score (e.g. very low) for missing docs
        default_score = float("-inf")
        vsm_scores = {did: default_score for did in bm25_ids}
        for doc in vsm_results:
            did = doc["doc_id"]
            if did in vsm_scores:
                vsm_scores[did] = doc["score"]

        # 3) Sort by VSM (descending)
        sorted_ids = sorted(bm25_ids, key=lambda did: vsm_scores[did], reverse=True)

        reranked = []
        for rank, did in enumerate(sorted_ids, start=1):
            reranked.append({
                "rank": rank,
                "doc_id": did,
                "bm25_score": bm25_scores[did],
                "vsm_score": vsm_scores[did],
                "parent": bm25_parents[did],
                "grandparent": bm25_grandparents[did],
            })

        return reranked


    # first vsm and then rerank based on bm25
    else: ## not working. a pain in the ass
        # 1. Run VSM to get a list of file-paths (or strings) of your top-K VSM docs:
        top_k_vsm = aux_vsm.run_word2vec_query(paths, query, top_k=100)
        vsm_ids = [r["doc_id"] for r in top_k_vsm["results"]]

        # 2. Feed those into BM25 for re-ranking:
        bm25_reranked = aux_bm25.run_bm25_query(
            paths,
            query,
            top_k=10,
            vsm_ids=vsm_ids
        )

        logging.info(f"-------------------Rerank {top_k} results for vsm + bm25-------------------")

        aux_bm25.print_documents(bm25_reranked, top_k=top_k)


def results_to_dataframe(
    document_list: List[Dict[str, Any]],
    ranking: str,
    existing_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Convert a list of document result dicts into a pandas DataFrame and
    optionally append to an existing DataFrame for iterative calls.

    Args:
        document_list: list of result dicts (as from retrieve functions)
        ranking: string label of ranking method used
        existing_df: optional DataFrame to append to

    Returns:
        A pandas DataFrame with columns:
          model, ranking, rank, grandparent, parent,
          bm25_score, vsm_score, combined_score,
          rrf_score, rrf_vsm, rrf_bm25
    """
    rows = []
    for entry in document_list:
        if 'rrf_score' in entry:
            model = 'rrf'
        elif 'combined_score' in entry:
            model = 'hybrid'
        elif 'bm25_score' in entry and 'vsm_score' in entry:
            model = 'bm25-vsm'
        else:
            model = 'unknown'

        rows.append({
            'model': model,
            'ranking': ranking,
            'rank': entry.get('rank'),
            'grandparent': entry.get('grandparent'),
            'parent': entry.get('parent'),
            'bm25_score': entry.get('bm25_score'),
            'vsm_score': entry.get('vsm_score'),
            'combined_score': entry.get('combined_score'),
            'rrf_score': entry.get('rrf_score'),
            'rrf_vsm': entry.get('rrf_vsm'),
            'rrf_bm25': entry.get('rrf_bm25')
        })
    new_df = pd.DataFrame(rows)
    if existing_df is not None:
        return pd.concat([existing_df, new_df], ignore_index=True)
    return new_df


if __name__ == '__main__':
    logger.info("Module loaded. Ready to merge.")