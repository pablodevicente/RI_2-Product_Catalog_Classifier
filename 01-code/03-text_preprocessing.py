"""
Using the github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
import os
import logging

import text_preprocessing as txtp


def concat_txt(folder_path):
    """
    Concatenates 'text.txt', 'tables.txt', and 'images_to_txt.txt' into a single 'pdf.txt'
    for each valid pdf folder in the given directory.

    Parameters:
    - folder_path (str): Path to the root directory containing labeled folders.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    for root, _, files in os.walk(folder_path):
        # Ensure we're only processing actual pdf folders
        if {"text.txt", "tables.txt", "images_to_txt.txt"}.issubset(set(files)):
            pdf_path = os.path.join(root, "pdf.txt")
            logging.debug(f"Creating {pdf_path}")

            try:
                # First write text.txt, then append tables.txt, then images_to_txt.txt
                with open(pdf_path, "w", encoding="utf-8") as pdf_file:
                    for txt_filename in ["text.txt", "tables.txt", "images_to_txt.txt"]:
                        txt_path = os.path.join(root, txt_filename)
                        if os.path.exists(txt_path):
                            with open(txt_path, "r", encoding="utf-8") as txt_file:
                                pdf_file.write(txt_file.read() + "\n")

                logging.info(f"Successfully created {pdf_path}")

                # Read and return the contents of the created PDF file
                with open(pdf_path, "r", encoding="utf-8") as pdf_file:
                    pdf_contents = pdf_file.read()
                return pdf_contents,pdf_path

            except Exception as e:
                logging.error(f"Error processing {root}: {e}")
                return None  # or handle the error as needed

def keep_alpha_numeric(input_text: str) -> str:
    """ Remove any character except alphanumeric characters and newlines """
    return ''.join(c for c in input_text if c.isalnum() or c in {' ', '\n'})


def process_nlp(file: str, path: str):
    """
    Applies NLP preprocessing to a given text and saves the processed output to a file.

    Parameters:
    - file (str): The input text to preprocess.
    - path (str): The output file path to save the processed text.
    """

    preprocess_functions = [
        txtp.to_lower, keep_alpha_numeric, txtp.remove_email,
        txtp.remove_phone_number, txtp.remove_itemized_bullet_and_numbering,
        txtp.remove_stopword, txtp.remove_url, txtp.remove_punctuation, txtp.lemmatize_word
    ]

    if not isinstance(file, str):
        raise ValueError(f"Expected file to be a string, but got {type(file)}")

    preprocessed_text = file

    for func in preprocess_functions:
        try:
            result = func(preprocessed_text)
            if isinstance(result, str) and result.strip():  # Ensure it's a valid non-empty string
                preprocessed_text = result
        except Exception as e:
            logging.DEBUG(f"Skipping {func.__name__} due to error: {e}")

    try:
        with open(path, "w", encoding="utf-8") as output_file:
            output_file.write(preprocessed_text)
        logging.info(f"Successfully saved preprocessed text to {path}")
    except Exception as e:
        logging.error(f"Error saving processed text: {e}")

def cleanup_txt_files(folder_path):
    """
    Deletes 'text.txt', 'tables.txt', and 'images_to_txt.txt' from each subfolder
    after they have been concatenated into 'pdf.txt'.

    Parameters:
    - folder_path (str): Path to the root directory containing labeled folders.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    for root, _, files in os.walk(folder_path):
        txt_files_to_delete = ["text.txt", "tables.txt", "images_to_txt.txt"]

        # Check if pdf.txt exists before deleting
        pdf_path = os.path.join(root, "pdf.txt")
        if os.path.exists(pdf_path):
            for txt_filename in txt_files_to_delete:
                txt_path = os.path.join(root, txt_filename)
                if os.path.exists(txt_path):
                    try:
                        os.remove(txt_path)
                        logging.info(f"Deleted: {txt_path}")
                    except Exception as e:
                        logging.error(f"Error deleting {txt_path}: {e}")


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
        if {"text.txt", "tables.txt", "images_to_txt.txt"}.issubset(set(files)):
            logging.info(f"Processing folder: {root}")

            try:
                pdf_file,pdf_path = concat_txt(root)  # Merge text files into pdf.txt
                process_nlp(pdf_file,pdf_path)  # Clean up pdf.txt
                logging.info(f"Successfully processed {root}")

            except Exception as e:
                logging.error(f"Error processing {root}: {e}")


# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

pdf_path = "../02-data/01-pdfs/accessories"
process_txt(pdf_path)
#cleanup_txt_files(pdf_path)