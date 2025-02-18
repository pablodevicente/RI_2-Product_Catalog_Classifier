from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Load pre-trained Word2Vec model
word2vec_model = Word2Vec.load('path_to_your_pretrained_word2vec_model')

# Function to tokenize and get word vectors
def get_word_vectors(text):
    tokens = word_tokenize(text.lower())  # Tokenize and lowercase
    word_vectors = []
    for token in tokens:
        if token in word2vec_model.wv:
            word_vectors.append(word2vec_model.wv[token])
    return word_vectors

# Read from pdf.txt
with open('pdf.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Get vectors for each word in the text
word_vectors = get_word_vectors(text)

# Example: Calculate document vector by averaging word vectors
if word_vectors:
    doc_vector = sum(word_vectors) / len(word_vectors)
    print("Document vector shape:", doc_vector.shape)
else:
    print("No word vectors found in the text.")

# You can further process `doc_vector` or store it as needed for your VSM application.


import os
import logging


def process_txt(folder_path):
    """
    Processes text files in labeled folders. For each folder containing
    ('text.txt', 'tables.txt', 'images_to_txt.txt'):

    1. Calls `concat_txt` to merge them into `pdf.txt`
    2. Calls `process_nlp_in_place` to clean up `pdf.txt`

    Parameters:
    - folder_path (str): Path to the root directory containing label folders.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    for root, _, files in os.walk(folder_path):
        if {"pdf.txt"}.issubset(set(files)):
            logging.info(f"Processing folder: {root}")

            try:

                logging.info(f"Successfully processed {root}")

            except Exception as e:
                logging.error(f"Error processing {root}: {e}")