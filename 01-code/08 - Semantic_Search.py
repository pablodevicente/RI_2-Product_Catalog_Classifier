import os
import numpy as np
import aux_vsm as aux
import fasttext.util
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
from gensim.models import KeyedVectors
import fasttext
import fasttext.util
from typing import List, Dict, Any, Tuple
# Split text into sentences (you can use nltk or a basic split)
import nltk

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


def filter_expansion_candidates(
    sims: List[Tuple[str, float]],
    idf: Dict[str, float],
    min_sim: float = 0.6,
    min_idf: float = 1.0
) -> List[str]:
    """
    Filter a list of (term, similarity) pairs by:
      1) similarity ≥ min_sim
      2) idf(term) ≥ min_idf (defaults to excluding OOV terms or very common words)

    Args:
        sims:       output of model.most_similar(token, topn=…)
        idf:        dict mapping term → IDF score
        min_sim:    minimum similarity score to keep
        min_idf:    minimum IDF score to keep

    Returns:
        filtered_terms: list of terms passing both thresholds
    """
    filtered_terms = []
    for term, sim in sims:
        if sim < min_sim:
            continue
        term_idf = idf.get(term, 0.0)
        if term_idf < min_idf:
            continue
        filtered_terms.append(term)
    return filtered_terms


def expand_query_terms(
    query: str,
    model: Any,
    idf_cached: Dict[str, float],
    tokenize_fn,
    topn_synonyms: int = 10,
    min_sim: float = 0.65,
    min_idf: float = 1.2
) -> List[str]:
    """
    For each token in `query`, find the top-n most similar terms in the model,
    filter them by similarity and IDF thresholds, and return a unique list
    of expansion terms.
    """
    tokens = tokenize_fn(query)
    expansions: List[str] = []

    for t in tokens:
        try:
            sims = model.most_similar(t, topn=topn_synonyms)
            filtered = filter_expansion_candidates(
                sims, idf_cached, min_sim, min_idf
            )
            expansions.extend(filtered)
        except KeyError:
            # token not in vocabulary
            continue

    # preserve order and dedupe
    return list(dict.fromkeys(expansions))


def embed_query(
    model: Any,
    query: str,
    idf_cached: Dict[str, float],
    tokenize_fn,
    topn_synonyms: int = 10,
    min_sim: float = 0.65,
    min_idf: float = 1.2
) -> Dict[str, Any]:
    """
    Expand a query by:
      1) original tokens
      2) word2vec synonyms (filtered by similarity and IDF)
      3) TF-IDF weighted average embedding

    Returns a dict containing the original query, its vector, and expansions.
    """
    vector_size = model.vector_size

    no_expansion = 0
    if no_expansion == 1:
        # 1. Expand query terms
        expansions = expand_query_terms(
            query=query,
            model=model,
            idf_cached=idf_cached,
            tokenize_fn=tokenize_fn,
            topn_synonyms=topn_synonyms,
            min_sim=min_sim,
            min_idf=min_idf,
        )

        # 2. Merge original tokens + expansions, preserving order
        original_tokens = tokenize_fn(query)
        all_tokens = list(dict.fromkeys(original_tokens + expansions))

    else:
        expansions = ["placeholder","shut-it"]
        original_tokens = tokenize_fn(query)
        all_tokens = list(dict.fromkeys(original_tokens))


    # 3. Compute TF-IDF weighted average embedding
    vector = aux.tfidf_weighted_avg_embedding(
        doc_tokens=all_tokens,
        model=model,
        idf_dict=idf_cached,
        vector_size=vector_size
    )

    return {
        "query": query,
        "vector": vector,
        "expansions": expansions,
    }


def retrieve_top_k_documents(
    query_vector: np.ndarray,
    corpus_vectors: Dict[str, np.ndarray],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """
    Ranks and retrieves top-k most similar documents based on cosine similarity.

    Args:
        query_vector (np.ndarray): The embedded query vector.
        corpus_vectors (Dict[str, np.ndarray]): Mapping from doc IDs to vectors.
        top_k (int): Number of top documents to retrieve.

    Returns:
        List[Tuple[str, float]]: List of (doc_id, similarity score), ranked.
    """
    doc_ids = list(corpus_vectors.keys())
    matrix = np.array([corpus_vectors[doc_id] for doc_id in doc_ids])

    similarities = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [(doc_ids[i], similarities[i]) for i in top_indices]


# Configure root logger once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    paths = {
        'word2vec': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'idf_cache': Path("../02-data/03-VSM/idf_cache_path.pkl"),
        'word2vec_vsm' : Path("../02-data/03-VSM/01-Word2Vec/word2vec-4-50-4-150.pkl"),
        'file' : Path("../02-data/00-testing/batteries-non-rechargable-primary/1cr2/1cr2.txt"),
        'output_path' : Path("../02-data/00-testing/batteries-non-rechargable-primary/1cr2/sentence_expansions.txt")
    }

    model = aux.load_word2vec_model(paths['word2vec'])
    texts = ["placeholder","data"]
    idf_dict = aux.get_or_build_idf(texts,str(paths['idf_cache']))

    file_path = paths["file"]
    text = file_path.read_text(encoding='utf-8')

    output_file = paths["output_path"]

    sentences = sent_tokenize(text)
    query = "cylindrical object with  metallic appearance the object has  diameter of approximately 10 mm and  length of about 20 mm"

    result = embed_query(
        model=model,
        query=query,
        idf_cached=idf_dict,
        tokenize_fn=aux.simple_tokenize
    )

    vsm_path = paths["word2vec_vsm"]
    with vsm_path.open("rb") as f:
        corpus_vectors: Dict[str, np.ndarray] = pickle.load(f)

    top_k_results = retrieve_top_k_documents(result["vector"],corpus_vectors,top_k=10)
    for doc_id, score in top_k_results:
        print(f"{doc_id:30s} → {score:.4f}")

