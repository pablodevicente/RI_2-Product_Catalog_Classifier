import os
import numpy as np
import gensim.downloader as api
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import KeyedVectors
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import fasttext
import fasttext.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler("processing.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

def process_fasttext(txt_filename, model):
    """
    Processes a text file and generates a document vector using FastText embeddings.

    Args:
        txt_filename (str): Path to the text file.
        model: A pre-trained FastText model.

    Returns:
        np.ndarray or None: A document vector (mean of word vectors) or None if processing fails.
    """

    if not os.path.exists(txt_filename):
        logging.warning(f"File not found: {txt_filename}")
        return None

    try:
        with open(txt_filename, "r", encoding="utf-8") as file:
            text = file.read()

        # Tokenize and filter out non-alphabetic words
        words = [word for word in text.split() if word.isalpha()]

        # Compute document vector as the mean of word vectors
        word_vectors = [model.get_word_vector(word) for word in words if word in model.words]

        if not word_vectors:
            logging.warning(f"No valid words found in {txt_filename}")
            return None

        doc_vector = np.mean(word_vectors, axis=0)  # Compute mean vector

        return doc_vector

    except Exception as e:
        logging.error(f"Error processing file {txt_filename}: {e}")
        return None


def process_pdf_directory(directory,model):
    """
    Iterates through a directory, processes text files corresponding to PDF folders,
    and computes document vectors.

    Args:
        directory (str): The root directory containing folders with PDFs.
        model : fassttext
    Returns:
        dict: A dictionary mapping folder paths to their document vectors.
    """


    corpus_vectors = {}
    logging.info(f"Processing directory: {directory}")

    for root, _, _ in os.walk(directory):

        file_name = os.path.basename(root)
        txt_filename = os.path.join(root, f"{file_name}.txt")

        doc_vector = process_fasttext(txt_filename,model) ## Returns vector for the document
        if doc_vector is not None:
            corpus_vectors[root] = doc_vector

    logging.info(f"Processed {len(corpus_vectors)} documents successfully.")
    return corpus_vectors


if __name__ == "__main__":

    fasttext.util.download_model('en', if_exists='ignore')  # Download pre-trained English FastText model
    ft_model = fasttext.load_model("cc.en.300.bin")

    pdfs_dir ='../02-data/00-testing/'
    output_path = "../02-data/03-VSM/fasttext-demo.pkl"

    # Create VSM from pdfs_dir
    document_vectors = process_pdf_directory(pdfs_dir,ft_model)

    with open(output_path, "wb") as f:
        pickle.dump(document_vectors, f)
    logging.info(f"Saved document vectors to {output_path}.")

