import os
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec

model = api.load("word2vec-google-news-300")  # Google's 300-dim Word2Vec
pdfs_dir = "../02-data/01-pdfs/accessories"
document_vectors = {}

def process_pdf_directory(directory):
    # Iterate through all subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "pdf.txt":
                file_path = os.path.join(root, file)

                # Read and preprocess the text file
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().split()

                # Convert words to vectors
                word_vectors = [model[word] for word in text if word in model]

                # Compute document vector (mean of word embeddings)
                if word_vectors:
                    doc_vector = np.mean(word_vectors, axis=0)
                else:
                    doc_vector = np.zeros(model.vector_size)  # Default to zero vector if no words are found

                # Store the document vector using the file path as the key
                document_vectors[file_path] = doc_vector

process_pdf_directory(pdfs_dir)

# Print summary
print(f"Processed {len(document_vectors)} documents.")
np.save("../02-data/00-testing/vsm1.npy", document_vectors)
