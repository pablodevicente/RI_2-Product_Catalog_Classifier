import os
import pickle
import numpy as np
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve, validation_curve

import matplotlib.pyplot as plt

# Configure logging
# Configure logging to both console and file
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(),  # Log to console
                        logging.FileHandler("../02-data/04-Classifier/preeliminary_classification.txt")  # Log to file
                    ])

def plot_learning_curve(name, best_model, X_train, y_train):
    """Plot the learning curve for the classifier."""
    train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, n_jobs=-1)

    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Test score')
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title(f'{name} Learning Curve')
    plt.legend()
    plt.show()

def plot_validation_curve(name, best_model, X_train, y_train):
    """Plot the validation curve for the classifier."""
    if 'C' not in best_model.get_params():
        logging.warning(f"{'C'} is not a valid parameter for {name}. Skipping validation curve plot.")
        return

    param_range = [0.1, 1, 10]
    train_scores, test_scores = validation_curve(best_model, X_train, y_train, param_name='C', param_range=param_range, cv=5)

    plt.plot(param_range, train_scores.mean(axis=1), label='Train score')
    plt.plot(param_range, test_scores.mean(axis=1), label='Test score')
    plt.xlabel('C Parameter')
    plt.ylabel('Accuracy')
    plt.title(f'{name} Validation Curve')
    plt.legend()
    plt.show()

def extract_category(file_path: str) -> str:
    """
    Extracts the category from the given file path.

    Args:
        file_path (str): The path to extract the category from.

    Returns:
        str: The extracted category.
    """
    normalized_path = os.path.normpath(file_path)
    parts = normalized_path.split(os.sep)
    if len(parts) < 4:
        logging.warning(f"Error extracting category from path: {file_path}")
        return ""
    return parts[-2]  # Extract the second-to-last folder as the category

def load_document_vectors(file_path: str):
    """Loads document vectors from a given pickle file."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None


def train_and_evaluate_classifiers(doc_vectors, doc_paths, file_name, results):
    """
    Trains and evaluates Logistic Regression, SVM, and Random Forest classifiers using GridSearchCV.

    Args:
        doc_vectors (dict): Dictionary of document vectors.
        doc_paths (list): List of document file paths.
        file_name (str): Name of the file being processed.
        results (list): List to store results.
    """
    # Convert document vectors to array and extract labels
    X = np.array(list(doc_vectors.values()))
    y_labels = [extract_category(doc) for doc in doc_paths]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Auxiliary Checks
    logging.info(f"Class distribution in training set: {dict(sorted(Counter(y_train).items()))}")
    logging.info(f"Class distribution in test set: {dict(sorted(Counter(y_test).items()))}")
    logging.info(f"Mean Cosine Similarity in Training Set: {cosine_similarity(X_train).mean():.4f}")

    param_grids = {
        "Logistic Regression": {
            'C': [0.1, 1, 10],
            'max_iter': [500, 1000]
        },
        "SVM": {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        },
        "Random Forest": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20]
        }
    }

    classifiers = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }

    for name, clf in classifiers.items():
        logging.info(f" ----------- Training {name} -----------")

        grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        cross_val_scores = cross_val_score(best_model, X_train, y_train, cv=5)

        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logging.info(f"{name} Best Params: {grid_search.best_params_}")
        logging.info(f"{name} Cross-validation Accuracy: {cross_val_scores.mean():.4f} ± {cross_val_scores.std():.4f}")
        logging.info(f"{name} Test Set Accuracy: {accuracy:.4f}")

        # Additional evaluations
        logging.info(f"{name} Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}")
        logging.info(
            f"{name} Classification Report: \n{classification_report(y_test, y_pred, target_names=label_encoder.classes_)}")

        # For ROC AUC (binary classification)
        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            logging.info(f"{name} ROC AUC: {roc_auc:.4f}")

        # Plot learning and validation curves
        plot_learning_curve(name, best_model, X_train, y_train)
        plot_validation_curve(name, best_model, X_train, y_train)

        result_record = {
            "file_name": file_name,
            "classifier": name,
            "best_params": grid_search.best_params_,
            "accuracy": accuracy,
            "train_class_distribution": dict(sorted(Counter(y_train).items())),
            "test_class_distribution": dict(sorted(Counter(y_test).items())),
            "mean_cosine_similarity_train": cosine_similarity(X_train).mean(),
            "cross_val_accuracy": cross_val_scores.mean(),
            "cross_val_std": cross_val_scores.std(),
            "confusion_matrix": confusion_matrix(y_test, y_pred),  # You may convert this if needed.
            "classification_report": classification_report(
                y_test, y_pred, target_names=label_encoder.classes_
            )
        }

        # Append the flat dictionary to your results list.
        results.append(result_record)

if __name__ == "__main__":
    files = ["../02-data/03-VSM/01-Word2Vec/word2vec-4-50-4-150.pkl",
             "../02-data/03-VSM/02-Glove/glove-4-50-4-150.pkl",
             "../02-data/03-VSM/03-Fasttext/fasttext-4-50-4-150.pkl"]

    save_path = "../02-data/04-Classifier/classifiers-4-50-4-150.csv"

    results = []

    for file_path in files:
        document_vectors = load_document_vectors(file_path)
        if document_vectors is not None:
            doc_paths = list(document_vectors.keys())
            logging.info(f"=========== Training with VSM {file_path} ===========")
            train_and_evaluate_classifiers(document_vectors, doc_paths, os.path.basename(file_path), results)

    df_results = pd.DataFrame(results)
    df_results.to_csv(save_path, index=False)
    logging.info(f"Results saved to {save_path}")

