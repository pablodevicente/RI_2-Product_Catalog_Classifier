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

def process_text_file(txt_path, model):
    """
    Reads a text file, tokenizes words, and computes its document vector using the provided word embedding model.

    Args:
        txt_path (str): Path to the text file.
        model: A pre-trained word embedding model (FastText or Word2Vec).

    Returns:
        np.ndarray: The computed document vector (mean of word embeddings).
                    Returns None if the file is not found or if an error occurs during processing.
    """

    if not os.path.exists(txt_path):
        logging.warning(f"File not found: {txt_path}, Skipping...")
        return None

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            words = f.read().split()

        # Extract word vectors for words present in the model
        word_vectors = []
        for word in words:
            if word in model:
                word_vectors.append(model[word])

        # Compute document vector (mean of word embeddings)
        doc_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
        logging.info(f"Processed: {txt_path} ({len(word_vectors)} valid words)") 

        return doc_vector

    except Exception as e:
        logging.error(f"Error processing {txt_path}: {e}")
        return None

def process_pdf_directory(directory,model):
    """
    Iterates through a directory, processes text files corresponding to PDF folders,
    and computes document vectors.

    Args:
        directory (str): The root directory containing folders with PDFs.
        model : A pre-trained word embedding model.
    Returns:
        dict: A dictionary mapping folder paths to their document vectors.
    """
    corpus_vectors = {}
    logging.info(f"Processing directory: {directory}")

    for root, _, _ in os.walk(directory):

        file_name = os.path.basename(root)
        txt_filename = os.path.join(root, f"{file_name}.txt")

        doc_vector = process_text_file(txt_filename,model) ## Returns vector for the document
        if doc_vector is not None:
            corpus_vectors[root] = doc_vector

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
    return process_pdf_directory(str(input_dir), model)


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