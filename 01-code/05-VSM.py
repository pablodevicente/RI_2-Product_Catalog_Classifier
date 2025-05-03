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
import fasttext.util
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def process_pdf_directory(
        directory: str,
        model: Any,
        tokenize_fn: callable,
        vector_size: int
) -> Dict[str, Any]:
    """
    Iterates through a directory, processes text files corresponding to PDF folders,
    and computes TF‑IDF‑weighted document vectors.

    Args:
        directory (str): The root directory containing folders with PDFs.
        model:        A pre-trained word embedding model or glove-index dict.
        tokenize_fn:  Function that takes raw text and returns a list of tokens.
        vector_size:  Dimensionality of the embedding vectors.

    Returns:
        dict: A mapping from folder paths to their TF‑IDF‑weighted document vectors.
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

    # 2. Fit TF‑IDF on the entire corpus
    logging.info("Fitting TF‑IDF vectorizer on the corpus...")
    _, idf_dict = aux.fit_tfidf(all_texts)

    # 3. Compute TF‑IDF‑weighted embeddings per document
    corpus_vectors: Dict[str, Any] = {}
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


def load_word2vec_model(google_path: Path) -> KeyedVectors:
    logger.info("Loading pre-trained Google News Word2Vec model...")
    return KeyedVectors.load_word2vec_format(str(google_path), binary=True)


def load_finetuned_word2vec(
    base_dir: Path, google_path: Path, finetuned_path: Path
) -> KeyedVectors:
    logger.info("Fine-tuning Word2Vec model...")
    aux.fine_tune_word2vec(str(base_dir), str(google_path), str(finetuned_path))
    logger.info("Loading fine-tuned Word2Vec model...")
    return KeyedVectors.load_word2vec_format(str(finetuned_path), binary=True)


def load_fasttext_model(fasttext_path: Path) -> Any:
    logger.info("Loading FastText model...")
    if not fasttext_path.exists():
        fasttext.util.download_model('en', if_exists='ignore')
        downloaded = Path("cc.en.300.bin")
        downloaded.rename(fasttext_path)
    return fasttext.load_model(str(fasttext_path))


def load_glove_index(glove_file: Path) -> Dict[str, Any]:
    logger.info("Loading GloVe embeddings...")
    return aux.load_glove_embeddings(str(glove_file))


def process_with_embedding_model(
    model: Union[KeyedVectors, Any], input_dir: Path
) -> Dict[str, Any]:
    logger.info(f"Processing directory {input_dir} with embedding model...")

    return process_pdf_directory(str(input_dir), model, aux.simple_tokenize, model.vector_size)


def process_with_glove(
    glove_index: Dict[str, Any], input_dir: Path
) -> Dict[str, Any]:
    logger.info(f"Processing directory {input_dir} with GloVe embeddings...")
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
    logger.info(f"Saved document vectors to {output_path}")


def main():
    args = parse_args()

    # Define model-specific paths
    paths = {
        'word2vec': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'word2vec_finetuned': Path("../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"),
        'fasttext': Path("../02-data/03-VSM/03-Fasttext/cc.en.300.bin"),
        'glove': Path("../02-data/03-VSM/02-Glove/glove.6B/glove.6B.100d.txt")
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
        'word2vec': lambda: load_word2vec_model(paths['word2vec']),
        'word2vec_finetuned':
            lambda: load_finetuned_word2vec(args.input_dir, paths['word2vec'], finetuned_path),
        'fasttext': lambda: load_fasttext_model(paths['fasttext']),
        'glove': lambda: load_glove_index(paths['glove'])
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
        vectors = process_with_embedding_model(model_or_index, args.input_dir)

    # Save results
    try:
        save_vectors(vectors, output_file)
    except Exception as e:
        logger.error(f"Failed to save vectors: {e}")


if __name__ == '__main__':
    main()