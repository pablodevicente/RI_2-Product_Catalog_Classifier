import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import aux_vsm
import aux_semantic_search

# Module-level logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def embed_query(
    model: Any,
    query: str,
    idf_cached: Dict[str, float],
    tokenize_fn: Any,
    use_expansion: bool = True,
    topn_synonyms: int = 10,
    min_sim: float = 0.65,
    min_idf: float = 1.2
) -> Dict[str, Any]:
    """
    Compute a TF-IDF weighted embedding for the query, optionally expanding terms.

    Args:
        model: Word2Vec-like model with attributes .vector_size.
        query: Raw query string.
        idf_cached: Mapping term -> IDF score.
        tokenize_fn: Function to tokenize the query string.
        use_expansion: Whether to include similar terms.
        topn_synonyms: Number of similar terms per token.
        min_sim: Similarity threshold for expansions.
        min_idf: IDF threshold for expansions.

    Returns:
        Dict containing:
          - query: original query string
          - vector: numpy array embedding
          - tokens: list of tokens used
          - expansions: list of expansion terms
    """
    # Tokenize original query
    original_tokens = tokenize_fn(query)
    tokens: List[str] = original_tokens.copy()
    expansions: List[str] = []

    if use_expansion:
        expansions = aux_semantic_search.expand_query_terms_vsm(
            original_tokens, model, idf_cached,
            topn_synonyms, min_sim, min_idf
        )
        # merge tokens and expansions, dedupe
        tokens = list(dict.fromkeys(original_tokens + expansions))

    # Compute TF-IDF weighted average embedding
    vector = aux_vsm.tfidf_weighted_avg_embedding(
        doc_tokens=tokens,
        model=model,
        idf_dict=idf_cached,
        vector_size=model.vector_size
    )

    logger.info("Query embedding computed: %d dims", vector.shape[0])
    return {
        "query": query,
        "vector": vector,
        "tokens": tokens,
        "expansions": expansions
    }


def retrieve_top_k_documents(
    query_vector: np.ndarray,
    corpus_vectors: Dict[str, np.ndarray],
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar document vectors by cosine similarity.

    Args:
        query_vector: Query embedding.
        corpus_vectors: Mapping doc_id -> embedding array.
        top_k: Number of results to return.

    Returns:
        List of dicts with keys: doc_id, score.
    """
    doc_ids = list(corpus_vectors.keys())
    matrix = np.vstack([corpus_vectors[i] for i in doc_ids])
    sims = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    results: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_idx, start=1):
        file_path = Path(doc_ids[idx])
        parent = file_path.parent.name
        grandparent = file_path.parent.parent.name
        results.append({
            "rank": rank,
            "doc_id": idx,
            "score": float(sims[idx]),
            "grandparent": grandparent,
            "parent": parent
        })
    logger.info("Top-%d documents retrieved", top_k)
    return results


def run_word2vec_query(
    paths: Dict[str, Path],
    query: str,
    top_k: int = 10,
    use_expansion: bool = True
) -> Dict[str, Any]:
    """
    Perform a Word2Vec-based semantic retrieval and return structured results.

    Args:
        paths: Dict containing keys:
            - 'word2vec': Path to binary model
            - 'idf_cache': Path for IDF cache
            - 'word2vec_vsm': Path to pickle of corpus vectors
            - 'pdf_folder': Path to corpus for building IDF
        query: Raw query string.
        top_k: Number of top documents to return.
        use_expansion: Whether to include expansion terms.

    Returns:
        List of result dicts with fields: rank, doc_id, score.
    """
    logger.info("Running Word2Vec query: '%s'", query)

    # Load resources
    model = aux_vsm.load_word2vec_model(paths['word2vec'])
    idf_dict = aux_vsm.get_or_build_idf(str(paths['pdf_folder']), str(paths['idf_cache']))

    with paths['word2vec_vsm'].open('rb') as f:
        corpus_vectors: Dict[str, np.ndarray] = pickle.load(f)

    # Embed query
    q_info = embed_query(
        model=model,
        query=query,
        idf_cached=idf_dict,
        tokenize_fn=aux_vsm.simple_tokenize,
        use_expansion=use_expansion
    )

    # Retrieve results
    results = retrieve_top_k_documents(q_info['vector'], corpus_vectors, top_k)

    # Wrap and return
    return {
        "query_info": q_info,
        "results": results
    }

def print_documents(top_k_vsm: Dict[str, Any], top_k: int = 5):
    """
    Expects top_k_vsm to be a dict with:
      - 'query_info': metadata about the query
      - 'results': List[Dict] of result items with keys 'rank', 'doc_id', 'score', etc.
    """
    logger.info(f"------------------- Showing top {top_k} VSM results -------------------")

    results = top_k_vsm["results"]
    for item in results[:top_k]:
        rank = item["rank"]
        score = item["score"]
        grandparent = item["grandparent"]
        parent = item["parent"]

        logger.info(f"Rank {rank} (score: {score:.2f}) - {grandparent} - {parent}")
    #        logger.info(f"Rank {rank} (score: {score:.2f}) - {grandparent} - {doc_id}")