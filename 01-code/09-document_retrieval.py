import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import bm25s
import aux_document_retrieval as aux

# Configure logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


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
    logger = logging.getLogger("retrieve_top_k_documents")
    try:
        doc_ids = list(corpus_vectors.keys())
        matrix = np.array([corpus_vectors[doc_id] for doc_id in doc_ids])

        similarities = cosine_similarity(query_vector.reshape(1, -1), matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [(doc_ids[i], float(similarities[i])) for i in top_indices]
        logger.info("Retrieved top %d documents successfully.", top_k)
        return results

    except Exception as e:
        logger.error("Error in retrieving top documents: %s", e, exc_info=True)
        return []

def load_corpus_bm25(corpus_path):

    corpus: List[str] = []
    file_names: List[str] = []

    for root, _, files in os.walk(corpus_path):
        parent = os.path.basename(root)
        target = f"{parent}.txt"
        if target in files:
            file_path = os.path.join(root, target)
            with open(file_path, encoding="utf-8") as f:
                text = f.read()
                corpus.append(text)
                file_names.append(str(file_path))

    return corpus,file_names

def create_bm25_index(input_path: Path, retriever_path: Path) -> bm25s.BM25:
    """
    Build and save a BM25 index from a corpus of documents.

    Args:
        input_path: Path to the folder containing PDF-derived text files.
        retriever_path: Path where the BM25 model and corpus will be saved.

    Returns:
        A fitted BM25 retriever instance.
    """
    logger = logging.getLogger("create_bm25_index")
    try:
        if not input_path.exists() or not input_path.is_dir():
            raise FileNotFoundError(f"Input path not found or is not a directory: {input_path}")

        logger.info("Loading corpus from %s", input_path)
        corpus,file_mapping = aux.load_corpus_bm25(input_path)

        logger.info("Tokenizing corpus")
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en")

        logger.info("Indexing corpus with BM25")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        retriever_path_parent = retriever_path.parent
        retriever_path_parent.mkdir(parents=True, exist_ok=True)

        retriever_path_corpus = Path(str(retriever_path) + "_corpus")

        logger.info("Saving BM25 model to %s", retriever_path)
        retriever.save(str(retriever_path))

        logger.info("Saving corpus to %s", retriever_path_corpus)
        retriever.save(str(retriever_path_corpus), corpus=corpus)

        with open(str(retriever_path) + "_filenames.json", "w", encoding="utf-8") as f:
            json.dump(file_mapping, f)

        logger.info("BM25 index creation complete.")
        return retriever

    except Exception as e:
        logger.error("Failed to create BM25 index: %s", e, exc_info=True)
        raise


def query_bm25(retriever_path, retriever: bm25s.BM25, query: str, k: int = 5, corpus=None):
    """
    Query the BM25 index and print top-k results.

    Args:
        retriever_path:
        retriever: A loaded BM25 model.
        query: The query string to search for.
        k: Number of top documents to retrieve.
        corpus: (Optional) Original corpus list for mapping IDs back to text.
    """
    logger = logging.getLogger("query_bm25")
    try:
        if not query:
            raise ValueError("Query string is empty.")

        logger.info("Tokenizing and querying BM25 for: '%s'", query)
        query_tokens = bm25s.tokenize(query, stopwords="en")

        doc_ids, scores = retriever.retrieve(query_tokens, k=k)

        with open(str(retriever_path) + "_filenames.json", "r", encoding="utf-8") as f:
            file_names = json.load(f)

        results = []
        for rank in range(doc_ids.shape[1]):
            doc_id = doc_ids[0, rank]
            score = scores[0, rank]
            doc_text = corpus[doc_id] if corpus is not None else None
            file_name = file_names[doc_id] if doc_id < len(file_names) else "Unknown file"
            results.append((rank + 1, doc_id, float(score), doc_text, file_name))

        logger.info("Query returned %d results.", len(results))
        for rank, doc_id, score, text, file_name in results:
            print(f"Rank {rank} (score: {score:.2f}): {file_name}")

    except Exception as e:
        logger.error("Error querying BM25: %s", e, exc_info=True)


if __name__ == "__main__":
    setup_logging()
    paths = {
        "pdf_folder": Path("../02-data/00-testing/"),
        "retriever": Path("../02-data/05-Retrieval/corpus_bm25")
    }

    try:
        if not paths["retriever"].exists():
            retriever = create_bm25_index(paths["pdf_folder"], paths["retriever"])
        else:
            logging.info("Loading existing BM25 model from %s", paths["retriever"])
            retriever = bm25s.BM25.load(str(paths["retriever"]), load_corpus=True)

        # Example query
        query_bm25(
            paths["retriever"],
            retriever,
            "this battery contains  positive temperature ",
            k=10,
            corpus=getattr(retriever, "corpus", None)
        )
    except Exception as error:
        logging.error("Application error: %s", error, exc_info=True)
