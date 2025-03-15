import os
import numpy as np
import pickle
import logging


# Load GloVe embeddings into a dictionary
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def create_vector_representation(file_path, embeddings_index):
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}, Skipping...")
        return None

    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        words = text.split()
        # Initialize an empty vector
        vector_representations = []
        for word in words:
            if word in embeddings_index:
                vector_representations.append(embeddings_index[word])

        # Aggregate vectors (e.g., average)
        if vector_representations:
            avg_vector = np.mean(vector_representations, axis=0)
            return avg_vector
        else:
            return None

def load_vectors_pickle(input_path):
    with open(input_path, 'rb') as f:
        return pickle.load(f)  # This will return a dictionary

def save_vectors_pickle(vectors_dict, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(vectors_dict, f)


def main():
    directory = '../02-data/00-testing/03-demo/'  # Input directory containing text files
    glove_file = '../02-data/03-VSM/02-Glove/glove.6B.100d.txt'  # Path to GloVe embeddings file
    output_pickle = '../02-data/03-VSM/02-Glove/glove-5.pkl'

    # Load GloVe embeddings
    logging.info("Loading GloVe embeddings...")
    embeddings_index = load_glove_embeddings(glove_file)

    # Process the directory and create vectors
    logging.info(f"Processing directory: {directory}")
    corpus_vectors = {}

    for root, _, _ in os.walk(directory):
        file_name = os.path.basename(root)
        txt_filename = os.path.join(root, f"{file_name}.txt")

        doc_vector = create_vector_representation(txt_filename, embeddings_index)

        if doc_vector is not None:
            corpus_vectors[root] = doc_vector
            logging.info(f"Vector representation for {root} created successfully.")
        else:
            logging.info(f"No valid GloVe vectors found for {txt_filename}")

    # Save vectors to pickle file
    save_vectors_pickle(corpus_vectors, output_pickle)

    # Log success message
    logging.info(f"Processed {len(corpus_vectors)} documents successfully.")
    logging.info(f"Saved document vectors to {output_pickle}.")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    main()
