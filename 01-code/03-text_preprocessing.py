"""
Using the github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
import os
import logging
import text_preprocessing as txtp

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def concat_txt(folder, required_files={"text.txt", "tables.txt", "images_to_txt.txt"}):
    """Concatenates contents of the found files into a single text file."""
    file_name = os.path.basename(folder)
    output_path = os.path.join(folder, f"{file_name}.txt")

    found_files = [f for f in required_files if os.path.exists(os.path.join(folder, f))]

    with open(output_path, "w", encoding="utf-8") as outfile:
        for filename in found_files:
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")  # Ensure newline separation

    return output_path

def keep_alpha_numeric(input_text: str) -> str:
    """ Remove any character except alphanumeric characters and newlines """
    return ''.join(c for c in input_text if c.isalnum() or c in {' ', '\n'})


def process_nlp(path: str):
    """
    Applies NLP preprocessing to a given text and saves the processed output to a file.

    Parameters:
    - path (str): The output file path to save the processed text.
    """

    preprocess_functions = [
        txtp.to_lower, keep_alpha_numeric, txtp.remove_email,
        txtp.remove_phone_number, txtp.remove_itemized_bullet_and_numbering,
        txtp.remove_stopword, txtp.remove_url, txtp.remove_punctuation, txtp.lemmatize_word
    ]

    if not isinstance(path, str):
        raise ValueError(f"Expected file to be a string, but got {type(path)}")

    with open(path, "r", encoding="utf-8") as file:
        preprocessed_text = file.read()

    for func in preprocess_functions:
        try:
            result = func(preprocessed_text)
            if isinstance(result, str) and result.strip():  # Ensure it's a valid non-empty string
                preprocessed_text = result
        except Exception as e:
            logging.info(f"Skipping {func.__name__} due to error: {e}")

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
        if required_files.intersection(set(files)):  # Check if at least one file is present
            logging.info(f"Processing folder: {root}")

            try:
                concat_file_path = concat_txt(root)
                process_nlp(concat_file_path)
                logging.info(f"Successfully processed {root}")

            except Exception as e:
                logging.error(f"Error processing {root}: {e}")



required_files = {"text.txt", "tables.txt", "images_to_txt.txt"}
pdf_path = "../02-data/00-testing/03-demo/"
process_txt(pdf_path)
#cleanup_txt_files(pdf_path)