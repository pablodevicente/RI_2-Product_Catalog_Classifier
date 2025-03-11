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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        #logging.FileHandler("processing.log"),  # Log to a file
        logging.StreamHandler()  # Log to console
    ]
)

def build_corpus(text_files_dir):
    nltk.download('punkt')  # Ensure tokenization models are downloaded

    sents = []  # List to store tokenized sentences

    # Traverse directory and process each text file
    for root, dirs, files in os.walk(text_files_dir):
        for file in files:
            if file.endswith(".txt"):  # Adjust as needed
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Tokenize text into sentences and then into words
                tokenized_sents = [word_tokenize(sent) for sent in sent_tokenize(text)]
                sents.extend(tokenized_sents)  # Add to the main list

    return sents

def fine_tune_word2vec(text_files_dir,pretrained_model_path, output_path):

    # build new vocab (from our own data)
    sentences = build_corpus(text_files_dir)

    pretrained_model = KeyedVectors.load_word2vec_format(
        pretrained_model_path, binary=True)
    pretrained_vocab = list(pretrained_model.index_to_key)

    # Create new model
    model = Word2Vec(vector_size=pretrained_model.vector_size)
    model.build_vocab(sentences)
    total_examples = model.corpus_count
    model_vocab = list(model.wv.index_to_key)

    # Load pretrained model's vocabulary.
    model.build_vocab([pretrained_vocab], update=True)

    # vectors_lockf property is initialize in __init__ method of Word2Vec class.
    # We are using build_vocab method to update vocabulary of model, so we need initialize vectors_lockf property manually.
    model.wv.vectors_lockf = np.ones(len(model.wv), dtype=np.float32)

    # load pretrained model's embeddings into model
    model.wv.intersect_word2vec_format(pretrained_model_path, binary=True)

    model.train(sentences,
                total_examples=total_examples, epochs=model.epochs)

    model.wv.save_word2vec_format(output_path, binary=True)


def process_text_file(txt_path,model):
    """
    Reads a text file, tokenizes words, and computes its document vector using Word2Vec.

    Args:
        txt_path (str): Path to the text file.

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


if __name__ == "__main__":

    finetune = 1

    if finetune == 0: #--------------------------- uses the normal version

        pdfs_dir ='../02-data/00-testing/'

        output_path = "../02-data/03-VSM/word2vec-demo-v2.pkl"
        word2vec_path = "word2vec-google-news-300"

        logging.info("Loading Google News Word2Vec KeyedVectors...")
        model = api.load(word2vec_path)  # Loads non-trainable KeyedVectors

        # Create VSM from pdfs_dir
        document_vectors = process_pdf_directory(pdfs_dir,model)

        with open(output_path, "wb") as f:
            pickle.dump(document_vectors, f)
        logging.info(f"Saved document vectors to {output_path}.")

    else: #--------------------------- uses the finetuned version

        text_files_dir = "../02-data/00-testing/" # where are the documents located?
        pretrained_model_path = "../02-data/03-VSM/word2vec-google-news-300.bin" #based model location
        model_path = "../02-data/03-VSM/word2vec_finetuned-v2.bin" # new model location
        vsm_path = "../02-data/03-VSM/word2vec-finetuned-demo-v2.pkl"

        fine_tune_word2vec(text_files_dir,pretrained_model_path, model_path)

        print("Loading Google News Word2Vec KeyedVectors finetuned...")
        model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        # Create VSM from pdfs_dir
        document_vectors = process_pdf_directory(text_files_dir,model)

        with open(vsm_path, "wb") as f:
            pickle.dump(document_vectors, f)
        logging.info(f"Saved document vectors to {vsm_path}.")