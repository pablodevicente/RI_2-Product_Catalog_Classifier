import os
import numpy as np
import gensim.downloader as api
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler("processing.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

# Load Google's 300-dim Word2Vec model
logging.info("Loading Word2Vec model...")
model = api.load("word2vec-google-news-300")
logging.info("Model loaded successfully.")


def process_text_file(txt_path):
    """
    Reads a text file, tokenizes words, and computes its document vector using Word2Vec.

    Args:
        txt_path (str): Path to the text file.

    Returns:
        np.ndarray: The computed document vector (mean of word embeddings).
                    Returns a zero vector if no words are found.
    """
    if not os.path.exists(txt_path):
        logging.warning(f"File not found: {txt_path}. Skipping...")
        return None

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            words = f.read().split()

        # Extract word vectors for words present in the model
        word_vectors = [model[word] for word in words if word in model]

        # Compute document vector (mean of word embeddings)
        doc_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)
        logging.info(f"Processed: {txt_path} ({len(word_vectors)} valid words)")
        return doc_vector

    except Exception as e:
        logging.error(f"Error processing {txt_path}: {e}")
        return None


def process_pdf_directory(directory):
    """
    Iterates through a directory, processes text files corresponding to PDF folders,
    and computes document vectors.

    Args:
        directory (str): The root directory containing folders with PDFs.

    Returns:
        dict: A dictionary mapping folder paths to their document vectors.
    """
    document_vectors = {}

    logging.info(f"Processing directory: {directory}")

    for root, _, _ in os.walk(directory):
        folder_name = os.path.basename(root)  # Folder name
        txt_filename = f"{folder_name}.txt"  # Expected text file name
        txt_path = os.path.join(root, txt_filename)

        doc_vector = process_text_file(txt_path)
        if doc_vector is not None:
            document_vectors[root] = doc_vector

    logging.info(f"Processed {len(document_vectors)} documents successfully.")
    return document_vectors


if __name__ == "__main__":
    pdfs_dir = "../02-data/00-testing/03-demo/"
    output_path = "../02-data/00-testing/vsm2.npy"

    document_vectors = process_pdf_directory(pdfs_dir)

    # Save document vectors
    np.save(output_path, document_vectors)
    logging.info(f"Saved document vectors to {output_path}.")
