import os
import numpy as np
import aux_vsm as aux
import fasttext.util
import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Union

from gensim.models import KeyedVectors
import fasttext
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def get_or_build_idf(
    texts: List[str],
    cache_path: str
) -> Dict[str, float]:
    """
    Returns an IDF dictionary, either by loading it from cache_path
    or by fitting on the provided texts and then caching it.

    Args:
        texts (List[str]): The corpus of raw documents.
        cache_path (str): Path to load/save the IDF dict.

    Returns:
        Dict[str, float]: Mapping from term to inverse document frequency.
    """
    if os.path.exists(cache_path):
        logging.info(f"Loading cached IDF dictionary from {cache_path}...")
        with open(cache_path, "rb") as f:
            idf_dict = pickle.load(f)
    else:
        logging.info("Fitting TF‑IDF vectorizer on the corpus...")
        _, idf_dict = aux.fit_tfidf(texts)
        logging.info(f"Caching IDF dictionary to {cache_path}...")
        with open(cache_path, "wb") as f:
            pickle.dump(idf_dict, f)
    return idf_dict


def process_pdf_directory(
    directory: str,
    model: Any,
    idf_cache_path: str,
    tokenize_fn: callable,
    vector_size: int,
) -> Dict[str, np.ndarray]:
    """
    Iterates through a directory, processes text files corresponding to PDF folders,
    and computes TF‑IDF‑weighted document vectors using a shared IDF.

    Args:
        directory (str): The root directory containing folders with PDFs.
        model: A pre-trained word embedding model or glove-index dict.
        tokenize_fn: Function that takes raw text and returns a list of tokens.
        vector_size: Dimensionality of the embedding vectors.
        idf_cache_path (str): Path to load/save the IDF dictionary.

    Returns:
        Dict[str, np.ndarray]: Mapping from folder paths (doc IDs) to their TF‑IDF‑weighted vectors.
    """
    logging.info(f"Scanning directory for corpus: {directory}")

    # 1. Gather all documents for TF‑IDF fitting
    all_texts: List[str] = []
    doc_paths: List[Tuple[str, str]] = []  # (doc_id, txt_path)
    for root, _, _ in os.walk(directory):
        file_name = os.path.basename(root)
        txt_path = os.path.join(root, f"{file_name}.txt")
        if os.path.isfile(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read()
            all_texts.append(text)
            doc_paths.append((root, txt_path))

    # 2. Build or load IDF dictionary
    idf_dict = get_or_build_idf(all_texts, idf_cache_path)

    # 3. Compute TF‑IDF‑weighted embeddings per document
    corpus_vectors: Dict[str, np.ndarray] = {}
    logging.info("Computing TF‑IDF‑weighted embeddings for each document...")
    for doc_id, txt_path in doc_paths:
        with open(txt_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        tokens = tokenize_fn(raw_text)

        vec = aux.tfidf_weighted_avg_embedding(
            doc_tokens=tokens,
            model=model,
            idf_dict=idf_dict,
            vector_size=vector_size
        )
        corpus_vectors[doc_id] = vec

    logging.info(f"Processed {len(corpus_vectors)} documents successfully.")
    return corpus_vectors

def process_with_embedding_model(
    model: Union[KeyedVectors, Any], input_dir: Path, idf_cache_path : Path
) -> Dict[str, Any]:
    logging.info(f"Processing directory {input_dir} with embedding model...")

    return process_pdf_directory(str(input_dir), model, str(idf_cache_path), aux.simple_tokenize, model.vector_size)



def process_with_glove(
    glove_index: Dict[str, Any], input_dir: Path
) -> Dict[str, Any]:
    logging.info(f"Processing directory {input_dir} with GloVe embeddings...")
    corpus_vectors: Dict[str, Any] = {}
    for txt_file in tqdm(input_dir.rglob("*.txt")):
        doc_id = str(txt_file.parent)
        vector = aux.create_vector_representation(str(txt_file), glove_index)
        if vector is not None:
            corpus_vectors[doc_id] = vector
    return corpus_vectors


def save_vectors(vectors: Dict[str, Any], output_path: Path) -> None:
    ensure_directories(output_path)
    with open(output_path, 'wb') as f:
        pickle.dump(vectors, f)
    logging.info(f"Saved document vectors to {output_path}")



# Configure root logger once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate document vectors using various embedding models."
    )
    parser.add_argument(
        "--model",
        choices=["word2vec", "word2vec_finetuned", "fasttext", "glove"],
        required=True,
        help="Embedding model to use."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../02-data/00-testing/"),
        help="Directory with text files to process."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../02-data/03-VSM/"),
        help="Base directory for saving vector pickles."
    )
    parser.add_argument(
        "--size",
        default="4-50-4-150",
        help="Descriptor tag for output filename."
    )
    return parser.parse_args()


def ensure_directories(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def main():
    args = parse_args()

    # Define model-specific paths
    paths = {
        'word2vec': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'word2vec_finetuned': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'fasttext': Path("../02-data/03-VSM/03-Fasttext/cc.en.300.bin"),
        'glove': Path("../02-data/03-VSM/02-Glove/glove.6B/glove.6B.100d.txt"),
        'idf_cache' : Path("../02-data/03-VSM/idf_cache_path.pkl")
    }
    finetuned_path = Path(
        "../02-data/03-VSM/01-Word2Vec/word2vec_finetuned-v2.bin"
    )

    # Determine output pickle path
    model_dir_map = {
        'word2vec': '01-Word2Vec',
        'word2vec_finetuned': '01-Word2Vec',
        'fasttext': '03-Fasttext',
        'glove': '02-Glove'
    }
    output_file = (
        args.output_dir
        / model_dir_map[args.model]
        / f"{args.model}-{args.size}.pkl"
    )

    # Model loading dispatch
    loaders = {
        'word2vec': lambda: aux.load_word2vec_model(paths['word2vec']),
        'word2vec_finetuned':
            lambda: aux.load_finetuned_word2vec(args.input_dir, paths['word2vec'], finetuned_path),
        'fasttext': lambda: aux.load_fasttext_model(paths['fasttext']),
        'glove': lambda: aux.load_glove_index(paths['glove'])
    }

    try:
        model_or_index = loaders[args.model]()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Processing dispatch
    if args.model == 'glove':
        vectors = process_with_glove(model_or_index, args.input_dir)
    else:
        vectors = process_with_embedding_model(model_or_index, args.input_dir, paths['idf_cache'])

    # Save results
    try:
        save_vectors(vectors, output_file)
    except Exception as e:
        logger.error(f"Failed to save vectors: {e}")


if __name__ == '__main__':
    main()