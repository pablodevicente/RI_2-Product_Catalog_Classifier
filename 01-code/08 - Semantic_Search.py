import os
import numpy as np
import aux_vsm as aux
import fasttext.util
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Union
from typing import Any, Callable, Dict, List, Union
import numpy as np
from gensim.models import KeyedVectors
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
from gensim.models import KeyedVectors
import fasttext
import fasttext.util
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any, Dict, Union


def embed_query(
    model,
    query: str,
    idf_cache: ,
    tokenize_fn,
    topn_synonyms: int = 10,
    min_sim: float = 0.65,
    min_idf: float = 1.2
) -> Dict[str, "vector"]:
    """
    Expand a query by:
      1) original tokens
      2) word2vec synonyms (filtered by similarity and IDF)
      3) TF-IDF weighted average embedding

    Returns a dict containing the original query and its vector.
    """
    # 1. Load IDF cache and model vector_size
    vector_size = model.vector_size

    # 2. Tokenize
    tokens = tokenize_fn(query)

    # 3. Gather and filter synonyms for each token
    expansions: List[str] = []
    for t in tokens:
        try:
            sims = model.most_similar(t, topn=topn_synonyms)
            filtered = filter_expansion_candidates(sims, idf_dict, min_sim, min_idf)
            expansions.extend(filtered)
        except KeyError:
            continue

    # 4. Merge original tokens + expansions
    all_tokens = list(dict.fromkeys(tokens + expansions))

    # 5. Compute TF-IDF weighted average embedding
    vector = aux.tfidf_weighted_avg_embedding(
        doc_tokens=all_tokens,
        model=model,
        idf_dict=idf_dict,
        vector_size=vector_size
    )

    return {"query": query, "vector": vector, "expansions": expansions}

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
        'word2vec': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin")
        'idf_cache': Path("../02-data/03-VSM/idf_cache_path.pkl")
        'file' : Path("../02-data/00-testing/batteries-non-rechargable-primary/1cr2")
    }
    model = aux.load_word2vec_model(paths['word2vec'])
    texts = ["placeholder","data"]
    idf_dict = aux.get_or_build_idf(texts,str(paths['idf_cache']))

    result = embed_query(
        model=model,
        query="battery of 500w",
        idf_cache=idf_dict,
        tokenize_fn=aux.simple_tokenize
    )
    print("Query:", result['query'])
    print("Vector shape:", result['vector'].shape)
    print("Expansions: ", result['expansions'])
