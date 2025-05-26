import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataclass import ChunkEntry,QueryEmbeddingResult,RetrievedDocument,TopKDocumentsResult,Word2VecQueryResult
import aux_vsm
import aux_semantic_search

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

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
) -> Word2VecQueryResult:
    """
    Ejecuta consulta Word2Vec usando recursos cargados, devolviendo objetos tipados.

    Args:
        resources: salida de load_word2vec_resources().
        query: cadena de consulta.
        top_k: número de documentos a recuperar.
        use_expansion: si se expande la query con sinónimos.

    Returns:
        Word2VecQueryResult con query_info y resultados tipados.
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

    topk_results = retrieve_top_k_documents(
        query_vector=q_info.vector,
        corpus_vectors=corpus_vectors,
        top_k=top_k
    )

    return Word2VecQueryResult(
        query_info=q_info,
        results=topk_results
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
) -> QueryEmbeddingResult:
    """
    Compute a TF-IDF weighted embedding for the query, optionally expanding terms,
    and devuelve un QueryEmbeddingResult.

    Args:
        model: Word2Vec-like model with .vector_size.
        query: Raw query string.
        idf_cached: Mapping term -> IDF score.
        tokenize_fn: Function to tokenize the query string.
        use_expansion: Si se añaden términos similares.
        topn_synonyms: Nº de sinónimos a extraer.
        min_sim: Umbral de similitud para expansiones.
        min_idf: Umbral de IDF para expansiones.

    Returns:
        QueryEmbeddingResult con todos los datos.
    """
    original_tokens = tokenize_fn(query)
    tokens = original_tokens.copy()
    expansions: List[str] = []

    if use_expansion:
        expansions = aux_semantic_search.expand_query_terms_vsm(
            original_tokens, model, idf_cached,
            topn_synonyms, min_sim, min_idf
        )
        # merge y dedupe
        tokens = list(dict.fromkeys(original_tokens + expansions))

    vector = aux_vsm.tfidf_weighted_avg_embedding(
        doc_tokens=tokens,
        model=model,
        idf_dict=idf_cached,
        vector_size=model.vector_size
    )

    logger.info("Query embedding computed: %d dims", vector.shape[0])

    return QueryEmbeddingResult(
        query=query,
        vector=vector,
        tokens=tokens,
        expansions=expansions
    )

def retrieve_top_k_documents(
    query_vector: np.ndarray,
    corpus_vectors: Dict[str, Union[np.ndarray, List[np.ndarray]]],  # key=path, value=vector o lista de vectores
    top_k: int = 10
) -> TopKDocumentsResult:
    """
    Recupera los top-k documentos más similares por similitud de coseno.
    Soporta representaciones por documento único o por chunks múltiples.

    Args:
        query_vector: Embedding de la consulta.
        corpus_vectors: Mapa path -> vector único o lista de vectors por chunks.
        top_k: número de documentos a devolver.

    Returns:
        TopKDocumentsResult con la lista de documentos ordenada.
    """
    multi_vector = any(
        isinstance(v, (list, tuple)) and len(v) > 1
        for v in corpus_vectors.values()
    )

    entries: List[ChunkEntry] = []
    for path, vec_or_list in corpus_vectors.items():
        p = Path(path)
        parent = p.name
        grandparent = p.parent.name
        if multi_vector and isinstance(vec_or_list, (list, tuple)):
            for vec in vec_or_list:
                entries.append(ChunkEntry(
                    doc_id=parent,
                    vector=vec,
                    parent=parent,
                    grandparent=grandparent,
                    full_path=str(p)
                ))
        else:
            ##because of bad coding and not wanting to re-do how i package the vectors (05-vsm)
            grandparent = p.parent.parent.name
            entries.append(ChunkEntry(
                doc_id=p.stem,
                vector=vec_or_list,
                parent=parent,
                grandparent=grandparent,
                full_path=str(p)
            ))

    matrix = np.vstack([e.vector for e in entries])
    sims = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
    for i, entry in enumerate(entries):
        entry.score = float(sims[i])

    best_per_doc: Dict[str, ChunkEntry] = {}
    for entry in entries:
        if (entry.doc_id not in best_per_doc
            or entry.score > best_per_doc[entry.doc_id].score):
            best_per_doc[entry.doc_id] = entry

    sorted_best = sorted(
        best_per_doc.values(),
        key=lambda e: e.score,
        reverse=True
    )[:top_k]

    documents = [
        RetrievedDocument(
            rank=idx+1,
            doc_id=entry.doc_id,
            score=entry.score,
            label=entry.grandparent
        )
        for idx, entry in enumerate(sorted_best)
    ]

    logger.info(f"Top-%d documents retrieved (multi-vector=%s)",
                top_k, multi_vector)
    return TopKDocumentsResult(
        top_k=top_k,
        multi_vector=multi_vector,
        documents=documents
    )

## desfasado -- refactor
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