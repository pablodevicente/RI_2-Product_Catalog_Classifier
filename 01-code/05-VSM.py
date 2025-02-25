import os
import numpy as np
import gensim.downloader as api
import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk.tokenize import word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler("processing.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)


def fine_tune_word2vec(existing_model_path, text_directory, new_model_path, vector_size=300, epochs=10):
    """
    Fine-tunes a pre-trained Word2Vec model by adding new words from text files.

    Args:
        existing_model_path (str): Path to the pre-trained Word2Vec model.
        text_directory (str): Directory containing new text files for training.
        new_model_path (str): Path to save the fine-tuned model.
        vector_size (int): Dimensionality of word vectors.
        epochs (int): Number of training epochs.

    Returns:
        None (Saves the fine-tuned model to `new_model_path`).
    """
    logging.info("Loading Google News Word2Vec KeyedVectors...")
    keyed_vectors = api.load("word2vec-google-news-300")  # Loads non-trainable KeyedVectors

    # Convert KeyedVectors into a trainable Word2Vec model
    logging.info("Converting KeyedVectors into a trainable Word2Vec model...")
    model = Word2Vec(vector_size=vector_size, min_count=1)
    model.build_vocab_from_freq({word: keyed_vectors.get_vecattr(word, "count") for word in keyed_vectors.index_to_key})
    model.wv = keyed_vectors  # Assign pre-trained vectors
    model.trainables.syn1neg = model.wv.vectors  # Assign negative sampling weights

    logging.info("Collecting new sentences from text files...")
    sentences = []
    for root, _, files in os.walk(text_directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
                    sentences.append(tokens)

    logging.info(f"Collected {len(sentences)} new sentences.")

    # Update vocabulary with new words
    logging.info("Updating model vocabulary with new words...")
    model.build_vocab(sentences, update=True)

    # Fine-tune the model
    logging.info(f"Training the model for {epochs} epochs...")
    model.train(sentences, total_examples=len(sentences), epochs=epochs)

    # Save the fine-tuned model
    logging.info(f"Saving fine-tuned model to {new_model_path}...")
    model.save(new_model_path)
    logging.info("Fine-tuning completed successfully.")

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

    existing_model_path = "word2vec-google-news-300"  # Path to existing model
    new_model_path = "../02-data/00-testing/word2vec_finetuned.model"  # Path to save the updated model

    fine_tune_word2vec(existing_model_path, pdfs_dir, new_model_path)

    #document_vectors = process_pdf_directory(pdfs_dir)

    # Save document vectors
    # np.save(output_path, document_vectors)
    logging.info(f"Saved document vectors to {output_path}.")
