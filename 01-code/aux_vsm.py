import os
import numpy as np
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

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
