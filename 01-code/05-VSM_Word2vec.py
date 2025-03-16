import os
import numpy as np
import gensim.downloader as api
import logging
from gensim.models import KeyedVectors
import pickle
import aux_vsm as aux
import fasttext
import fasttext.util

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
        model : Word2Vec model.
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

def main():
    """
    Main function to process document vectors using either a pre-trained or fine-tuned Word2Vec model.

    The script performs the following:
    1. If `finetune` is 0, it loads the pre-trained Google News Word2Vec model.
    2. If `finetune` is 1, it fine-tunes the Word2Vec model using the given text dataset.
    3. It processes text files and computes document vectors.
    4. The computed vectors are saved to disk for later use.

    Logging is used throughout to track progress and potential issues.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    pdfs_dir = "../02-data/00-testing/"

    word2vec_path = "../02-data/03-VSM/01-Word2Vec/word2vec-google-news-300.bin"
    finetuned_model_path = "../02-data/03-VSM/01-Word2Vec/word2vec_finetuned-v2.bin"

    vsm_out_path = "../02-data/03-VSM/01-Word2Vec/fasttext-5.pkl"

    model_choice = input("Choose model type (word2vec/fasttext): ").strip().lower()

    if model_choice == "word2vec":
        logging.info("Using pre-trained Google News Word2Vec model.")

        try:
            logging.info("Loading Google News Word2Vec KeyedVectors...")
            model = api.load(word2vec_path)  # Loads non-trainable KeyedVectors
            logging.info("Word2Vec model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Word2Vec model: {e}")
            return

    elif model_choice == "word2vec_finetuned":
        logging.info("Using fine-tuned Word2Vec model.")
        try:
            logging.info("Fine-tuning Word2Vec model...")
            aux.fine_tune_word2vec(pdfs_dir, word2vec_path, finetuned_model_path)
            logging.info("Fine-tuning complete.")

            logging.info("Loading fine-tuned Word2Vec model...")
            model = KeyedVectors.load_word2vec_format(finetuned_model_path, binary=True)
            logging.info("Fine-tuned Word2Vec model loaded successfully.")
        except Exception as e:
            logging.error(f"Error during fine-tuning or model loading: {e}")
            return

    elif model_choice == "fasttext":
        logging.info("Downloading and loading pre-trained FastText model.")
        try:
            fasttext.util.download_model('en', if_exists='ignore')
            model = fasttext.load_model("cc.en.300.bin")
            logging.info("FastText model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading FastText model: {e}")
            return
    else:
        logging.error("Invalid model type. Choose either 'word2vec' or 'fasttext'.")
        return

    try:
        logging.info(f"Processing text files in directory: {pdfs_dir}")
        document_vectors = process_pdf_directory(pdfs_dir, model)
        logging.info(f"Processed {len(document_vectors)} documents successfully.")

        # Save document vectors
        with open(vsm_out_path, "wb") as f:
            pickle.dump(document_vectors, f)
        logging.info(f"Saved document vectors to {vsm_out_path}.")

    except Exception as e:
        logging.error(f"Error during document vector processing: {e}")


if __name__ == "__main__":
    main()