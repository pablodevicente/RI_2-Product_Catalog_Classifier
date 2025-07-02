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
import yaml

import os
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np


def chunk_tokens(
        tokens: List[str],
        min_chunk_size: int = 600,
        max_chunks: int = 5,
        overlap_ratio: float = 0.5
) -> List[List[str]]:
    """
    Split `tokens` into up to `max_chunks` overlapping chunks of at least `min_chunk_size`.
    Overlap between chunks is controlled by `overlap_ratio` (e.g. 0.5 for 50% overlap).
    """
    total = len(tokens)
    if total == 0:
        return []

    # If document is short, return it as a single chunk
    if total <= min_chunk_size:
        return [tokens]

    stride = int(min_chunk_size * (1 - overlap_ratio))
    chunks: List[List[str]] = []
    start = 0

    while start < total and len(chunks) < max_chunks:
        end = min(start + min_chunk_size, total)
        chunks.append(tokens[start:end])
        logging.debug(f"Created chunk {len(chunks)}: tokens[{start}:{end}]")
        start += stride

    return chunks


def build_chunked_corpus_vectors(
        directory: Path,
        model: Any,
        idf_cache_path: str,
        tokenize_fn: callable,
        vector_size: int
) -> Dict[str, List[np.ndarray]]:
    """
    Walk through `directory` recursively. For each folder named XXX containing XXX.txt,
    read and tokenize the text, split into overlapping chunks, embed each chunk,
    and store all chunk vectors under the folder name.

    Returns:
        corpus_vectors: mapping from document ID (folder name) to list of chunk vectors.
    """
    corpus_vectors: Dict[str, List[np.ndarray]] = {}
    idf_dict = aux.get_or_build_idf(str(directory),idf_cache_path)

    for root, _, files in os.walk(directory):
        folder = Path(root).name
        label_dir = Path(root).parent.name
        folder_struct = os.path.join(label_dir, folder)

        txt_filename = f"{folder}.txt"

        if txt_filename not in files:
            continue  # no matching .txt here

        txt_path = Path(root) / txt_filename
        try:
            logging.debug(f"Processing document '{folder}' at {txt_path}")
            raw_text = txt_path.read_text(encoding="utf-8")
        except Exception as e:
            logging.warning(f"Could not read {txt_path}: {e}")
            continue

        # Tokenize
        try:
            tokens = tokenize_fn(raw_text)
            logging.debug(f"Tokenized '{folder}': {len(tokens)} tokens")
        except Exception as e:
            logging.warning(f"Tokenization failed for {txt_path}: {e}")
            continue

        # Split into overlapping chunks
        chunks = chunk_tokens(tokens)
        if not chunks:
            logging.warning(f"No chunks generated for {folder}, skipping.")
            continue
        logging.debug(f"Generated {len(chunks)} chunks for '{folder}'")

        # Embed each chunk
        vectors: List[np.ndarray] = []
        for idx, chunk in enumerate(chunks, start=1):
            try:
                vec = aux.tfidf_weighted_avg_embedding(
                    doc_tokens=chunk,
                    model=model,
                    idf_dict=idf_dict,
                    vector_size=vector_size
                )
                vectors.append(vec)
                logging.debug(f"Embedded chunk {idx} of '{folder}'")
            except Exception as e:
                logging.error(f"Embedding failed for chunk {idx} of {folder}: {e}")

        if vectors:
            corpus_vectors[folder_struct] = vectors
            logging.debug(f"Stored {len(vectors)} vectors for document '{folder}'")
        else:
            logging.warning(f"No vectors stored for '{folder}' due to embedding errors")

    return corpus_vectors


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

    # 1. Build or load IDF dictionary
    idf_dict = aux.get_or_build_idf(directory,idf_cache_path)

    # 2. Compute TF‑IDF‑weighted embeddings per document
    corpus_vectors: Dict[str, np.ndarray] = {}
    logging.info("Computing TF‑IDF‑weighted embeddings for each document...")

    for root, dirs, files in os.walk(directory):
        folder = Path(root)
        parent = folder.name
        grandparent = folder.parent.name
        expected_txt = f"{parent}.txt"

        if expected_txt in files:
            txt_path = Path(root) / expected_txt
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    raw_text = f.read()

                tokens = tokenize_fn(raw_text)
                logging.debug(f"File {txt_path} contains {len(tokens)} tokens")

                vec = aux.tfidf_weighted_avg_embedding(
                    doc_tokens=tokens,
                    model=model,
                    idf_dict=idf_dict,
                    vector_size=vector_size
                )

                key = f"{grandparent}/{parent}/{expected_txt}"
                corpus_vectors[key] = vec

            except Exception as e:
                logging.warning(f"Failed processing {txt_path}: {e}")

    return corpus_vectors


def process_with_embedding_model(model: Union[KeyedVectors, Any], input_dir: Path, idf_cache_path : Path, chunks : int) -> Dict[str, Any]:

    logging.info(f"Processing directory {input_dir} with embedding model...")

    if chunks == 0:
        return process_pdf_directory(str(input_dir), model, str(idf_cache_path), aux.simple_tokenize, model.vector_size)
    else:
        return build_chunked_corpus_vectors(input_dir, model, str(idf_cache_path), aux.simple_tokenize, model.vector_size)


## This could be refactored into the previous function -- more clean .-- dont wanna do it, i am not using glove in the end so...
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

def save_vectors(vectors: Dict[str, Any], output_path: Path, input_dir: Path) -> None:
    """
    Save a dict of document vectors, but rewrite each key k so that
    k is the full, absolute filesystem path to the document.
    """
    ensure_directories(output_path)

    new_vectors: Dict[str, Any] = {}
    for key, vec in vectors.items():
        full_path = os.path.join(input_dir,key)
        new_vectors[full_path] = vec
        logging.debug(f"Saved vector for document '{key}' with path {full_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(new_vectors, f)

    logging.info(f"Saved document vectors to {output_path}")

def ensure_directories(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

# Configure root logger once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_initial_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yml"),
        help="Path to YAML config file."
    )
    return parser.parse_known_args()[0]


def parse_args(defaults: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate document vectors using various embedding models.",
        parents=[argparse.ArgumentParser(add_help=False)]
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
        default=Path(defaults['input_dir']),
        help="Directory with text files to process."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(defaults['output_dir']),
        help="Base directory for saving vector pickles."
    )
    parser.add_argument(
        "--size",
        default=defaults['size'],
        help="Descriptor tag for output filename."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=defaults.get('config', Path('config.yml')),
        help="Path to YAML config file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve paths from config
    paths = cfg['paths']
    model_dirs = cfg['model_dirs']
    chunks = cfg.get('defaults', {}).get('chunks', 1)

    # Build full Path objects
    paths = {k: Path(v) for k, v in paths.items()}

    # Determine output path
    output_file = (
        args.output_dir
        / model_dirs[args.model]
        / f"{args.model}-{args.size}-{chunks}.pkl"
    )

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Dispatch loaders
    loaders = {
        'word2vec': lambda: aux.load_word2vec_model(paths['word2vec']),
        'word2vec_finetuned': lambda: aux.load_finetuned_word2vec(
            args.input_dir,
            paths['word2vec_finetuned_base'],
            paths['word2vec_finetuned_v2']
        ),
        'fasttext': lambda: aux.load_fasttext_model(paths['fasttext']),
        'glove': lambda: aux.load_glove_index(paths['glove'])
    }

    # Load models
    try:
        model_or_index = loaders[args.model]()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Processing dispatch -- should refactor this
    if args.model == 'glove':
        vectors = process_with_glove(model_or_index, args.input_dir)
    else:
        vectors = process_with_embedding_model(model_or_index, args.input_dir, paths['idf_cache'],chunks)

    # Save results
    try:
        save_vectors(vectors, output_file,args.input_dir)
    except Exception as e:
        logger.error(f"Failed to save vectors: {e}")


if __name__ == '__main__':
    main()