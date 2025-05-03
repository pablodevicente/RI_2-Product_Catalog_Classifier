import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any
from pathlib import Path

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


## tf-idf functions

def load_all_texts(input_dir: Path):
    """
    Loads all .txt files from the given directory and its subdirectories.

    Args:
        input_dir (Path): The root directory containing subfolders with .txt files.

    Returns:
        List[Tuple[str, str]]: A list of tuples where each tuple is (file_path, file_content).
    """
    text_data = []
    for txt_path in input_dir.rglob("*.txt"):
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                content = f.read()
                text_data.append((str(txt_path), content))
        except Exception as e:
            print(f"Failed to read {txt_path}: {e}")

    return text_data

def fit_tfidf(corpus: List[str]) -> (TfidfVectorizer, Dict[str, float]):
    """
    Fit a TfidfVectorizer on the full corpus and return both
    the fitted vectorizer and a word->idf lookup dict.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    idf_values     = vectorizer.idf_
    idf_dict = {word: idf_values[idx] for idx, word in enumerate(feature_names)}
    return vectorizer, idf_dict

def tfidf_weighted_avg_embedding(
    doc_tokens: List[str],
    model: Any,
    idf_dict: Dict[str, float],
    vector_size: int
) -> np.ndarray:
    """
    Compute the TF‑IDF‑weighted average embedding for a single document.
    - doc_tokens: list of str tokens for this document
    - model: any embedding model with __getitem__ or .wv[word] interface
    - idf_dict: word -> idf weight mapping
    - vector_size: dimensionality of your embeddings
    """
    weighted_sum = np.zeros(vector_size, dtype=float)
    weight_sum   = 0.0

    for word in doc_tokens:
        # handle both Gensim KeyedVectors and dict-based embeddings
        try:
            vec = (
                model.wv[word]   # for Word2Vec
                if hasattr(model, "wv")
                else model[word] # for FastText or glove-index dict
            )
        except KeyError:
            continue

        if word in idf_dict:
            w = idf_dict[word]
            weighted_sum += vec * w
            weight_sum   += w

    if weight_sum > 0:
        return weighted_sum / weight_sum
    else:
        # fallback to zero vector if no overlap
        return np.zeros(vector_size, dtype=float)

def simple_tokenize(text: str) -> List[str]:
    """
    Tokenizes input text into a list of lowercase words, stripping punctuation.

    Args:
        text (str): The raw text input.

    Returns:
        List[str]: List of clean tokens.
    """
    from nltk.tokenize import word_tokenize

    tokens = word_tokenize(text)  # handles punctuation and contractions
    # Remove punctuation and lower-case the tokens
    tokens = [token.lower() for token in tokens if token.isalpha()]

    return tokens
