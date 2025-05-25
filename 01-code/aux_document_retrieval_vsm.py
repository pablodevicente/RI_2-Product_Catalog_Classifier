import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass, field
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

@dataclass
class ChunkEntry:
    doc_id: str
    full_path: str
    vector: np.ndarray
    parent: str
    grandparent: str
    score: float = field(default=None)
    idx: int = field(default=None)

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
    corpus_vectors: Dict[str, Union[np.ndarray, List[np.ndarray]]],  # key=path, value=vector or list of chunk vectors
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar documents by cosine similarity.
    Supports both single-vector (one embedding per document) and
    multi-vector (list of chunk embeddings per document) setups.

    Returns list of dicts with keys:
      - rank
      - doc_id       (document folder name)
      - score        (max cosine similarity)
      - parent       (folder name)
      - grandparent  (folder's parent)
    """
    # Detect multi-vector: any value is a list/tuple with multiple vectors
    multi_vector = any(isinstance(v, (list, tuple)) and len(v) > 1 for v in corpus_vectors.values())

    # Build chunk entries with metadata
    entries: List[ChunkEntry] = []
    for path, vec_or_list in corpus_vectors.items():
        p = Path(path)
        parent = p.name
        grandparent = p.parent.name
        if multi_vector and isinstance(vec_or_list, (list, tuple)):
            # multiple chunk vectors under one document
            for i, vec in enumerate(vec_or_list):
                entries.append(ChunkEntry(
                    doc_id=parent,
                    full_path=path,
                    vector=vec,
                    parent=parent,
                    grandparent=grandparent,
                    idx=i
                ))
        else:
            # single vector per document
            doc_id = p.stem
            entries.append(ChunkEntry(
                doc_id=doc_id,
                full_path=path,
                vector=vec_or_list,
                parent=parent,
                grandparent=grandparent,
                idx=0
            ))

    # Compute cosine similarities for all entries
    matrix = np.vstack([e.vector for e in entries])
    sims = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]

    for i, entry in enumerate(entries):
        entry.score = float(sims[i])

    # Aggregate best chunk per document
    best_per_doc: Dict[str, ChunkEntry] = {}
    for entry in entries:
        if entry.doc_id not in best_per_doc or entry.score > best_per_doc[entry.doc_id].score:
            best_per_doc[entry.doc_id] = entry

    top_docs = sorted(
        best_per_doc.values(),
        key=lambda e: e.score,
        reverse=True
    )[:top_k]

    results: List[Dict[str, Any]] = []
    for rank, entry in enumerate(top_docs, start=1):
        results.append({
            "rank": rank,
            "doc_id": entry.doc_id,
            "score": entry.score,
            "parent": entry.parent,
            "grandparent": entry.grandparent
        })

    logger.info(f"Top-%d documents retrieved (multi-vector={multi_vector})", top_k)
    return results

def load_word2vec_resources(
    paths: Dict[str, Path],
    use_multivector: bool = True
) -> Dict[str, Any]:
    """
    Load model, idf, and corpus_vectors once.
    Returns a dict with keys 'model', 'idf', and 'corpus_vectors'.
    """
    model = aux_vsm.load_word2vec_model(paths['word2vec'])
    idf_dict = aux_vsm.get_or_build_idf(
        str(paths['pdf_folder']),
        str(paths['idf_cache'])
    )
    vec_key = 'word2vec_vsm_multivector' if use_multivector else 'word2vec_vsm_singlevector'
    with paths[vec_key].open('rb') as f:
        corpus_vectors: Dict[str, np.ndarray] = pickle.load(f)

    return {
        'model': model,
        'idf': idf_dict,
        'corpus_vectors': corpus_vectors
    }

def run_word2vec_query_preloaded(
    resources: Dict[str, Any],
    query: str,
    top_k: int = 10,
    use_expansion: bool = True
) -> Dict[str, Any]:
    """
    resources: output of load_word2vec_resources()
    Just embed & score — no file I/O.
    """
    model = resources['model']
    idf_dict = resources['idf']
    corpus_vectors = resources['corpus_vectors']

    q_info = embed_query(
        model=model,
        query=query,
        idf_cached=idf_dict,
        tokenize_fn=aux_vsm.simple_tokenize,
        use_expansion=use_expansion
    )

    results = retrieve_top_k_documents(q_info['vector'], corpus_vectors, top_k)

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