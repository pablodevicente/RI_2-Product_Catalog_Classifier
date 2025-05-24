import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def filter_expansion_candidates(
    sims: List[Tuple[str, float]],
    idf: Dict[str, float],
    min_sim: float = 0.8,
    min_idf: float = 1.0
) -> List[str]:
    """
    Filter a list of (term, similarity) pairs by similarity and IDF thresholds.

    Args:
        sims: List of (term, similarity) tuples from word2vec.
        idf: Mapping term -> IDF score.
        min_sim: Minimum similarity score to keep.
        min_idf: Minimum IDF score to keep.

    Returns:
        List of terms passing both thresholds.
    """
    filtered: List[str] = []
    for term, sim in sims:
        if sim < min_sim:
            continue
        if idf.get(term, 0.0) < min_idf:
            continue
        filtered.append(term)
    return filtered


def expand_query_terms_vsm(
    query_tokens: List[str],
    model: Any,
    idf: Dict[str, float],
    topn_synonyms: int = 10,
    min_sim: float = 0.65,
    min_idf: float = 1.2
) -> List[str]:
    """
    For each input token, find and filter top-n similar terms from the model.

    Args:
        query_tokens: List of tokens from the query.
        model: Word2Vec-like model with .most_similar()
        idf: Mapping term -> IDF score.
        topn_synonyms: Number of similar terms to retrieve per token.
        min_sim: Minimum similarity threshold.
        min_idf: Minimum IDF threshold.

    Returns:
        Unique list of expansion terms preserving order.
    """
    expansions: List[str] = []
    for token in query_tokens:
        try:
            sims = model.most_similar(token, topn=topn_synonyms)
            filtered = filter_expansion_candidates(sims, idf, min_sim, min_idf)
            expansions.extend(filtered)
        except KeyError:
            logger.debug("Token '%s' not in model vocabulary", token)
            continue
    # Dedupe while preserving order
    return list(dict.fromkeys(expansions))
