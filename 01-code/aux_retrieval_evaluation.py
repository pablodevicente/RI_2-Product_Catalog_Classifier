import pandas as pd
import logging
from typing import Dict, Tuple, List
from pathlib import Path
from dataclasses import dataclass
import aux_document_retrieval_bm25 as aux_bm25
import aux_document_retrieval_vsm as aux_vsm
import aux_semantic_search as aux_semantics

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