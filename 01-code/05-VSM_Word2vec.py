import os
import numpy as np
import gensim.downloader as api
import logging
from gensim.models import KeyedVectors
import pickle
import aux_vsm as aux

def process_text_file(txt_path,model):
    """
    Reads a text file, tokenizes words, and computes its document vector using Word2Vec.

    Args:
        txt_path (str): Path to the text file.
        model : Model to utilize for vectorization

    Returns:
        np.ndarray: The computed document vector (mean of word embeddings).
                    Returns a zero vector if no words are found.
    """

    if not os.path.exists(txt_path):
        logging.warning(f"File not found: {txt_path}, Skipping...")
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

    finetune = 0  # Set to 0 for pre-trained Word2Vec, 1 for fine-tuning

    if finetune == 0:
        logging.info("Using pre-trained Google News Word2Vec model.")

        # Define input/output paths
        pdfs_dir = "../02-data/00-testing/"
        word2vec_path = "../02-data/03-VSM/word2vec-google-news-300.bin"
        vsm_out_path = "../02-data/03-VSM/word2vec-demo-v2.pkl"

        try:
            logging.info("Loading Google News Word2Vec KeyedVectors...")
            model = api.load(word2vec_path)  # Loads non-trainable KeyedVectors
            logging.info("Word2Vec model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Word2Vec model: {e}")
            return

    else:
        logging.info("Using fine-tuned Word2Vec model.")

        # Define input/output paths for fine-tuning
        pdfs_dir = "../02-data/00-testing/"
        pretrained_model_path = "../02-data/03-VSM/word2vec-google-news-300.bin"
        finetuned_model_path = "../02-data/03-VSM/word2vec_finetuned-v2.bin"
        vsm_out_path = "../02-data/03-VSM/word2vec-finetuned-demo-v2.pkl"

        try:
            logging.info("Fine-tuning Word2Vec model...")
            aux.fine_tune_word2vec(pdfs_dir, pretrained_model_path, finetuned_model_path)
            logging.info("Fine-tuning complete.")

            logging.info("Loading fine-tuned Word2Vec model...")
            model = KeyedVectors.load_word2vec_format(finetuned_model_path, binary=True)
            logging.info("Fine-tuned Word2Vec model loaded successfully.")
        except Exception as e:
            logging.error(f"Error during fine-tuning or model loading: {e}")
            return

    # Process text files and generate document vectors
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