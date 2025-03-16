"""
Using the github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
import os
import logging
import text_preprocessing as txtp
import re
import argparse


def concat_txt(folder, required_files):
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

def remove_isolated_letters(text: str):
    """
    Removes isolated single letters from the text.

    Parameters:
    - text (str): The input text.

    Returns:
    - str: The cleaned text without single-letter words.
    """
    return re.sub(r'\b[a-zA-Z]\b', '', text)  # Removes standalone letters


def remove_random_numbers(text: str):
    """
    Removes lines that are entirely or mostly numbers while keeping numbers in valid sentences.

    Parameters:
    - text (str): The input text.

    Returns:
    - str: The cleaned text without isolated number sequences.
    """
    lines = text.split("\n")  # Split text into lines

    filtered_lines = []
    for line in lines:
        # Remove if the line is entirely numbers or mostly numbers with spaces
        if re.fullmatch(r'[\d\s]+', line):  # Matches lines with only digits and spaces
            continue

        filtered_lines.append(line)  # Keep valid lines

    return "\n".join(filtered_lines)  # Reconstruct text

def process_nlp(path: str):
    """
    Applies NLP preprocessing to a given text and saves the processed output to a file.

    Parameters:
    - path (str): The output file path to save the processed text.
    """

    preprocess_functions = [
        txtp.to_lower, keep_alpha_numeric, txtp.remove_email,
        txtp.remove_phone_number, txtp.remove_itemized_bullet_and_numbering,
        txtp.remove_stopword, txtp.remove_url, txtp.remove_punctuation, txtp.lemmatize_word,
        remove_isolated_letters, remove_random_numbers
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

def cleanup_txt_files(concat_file_path, folder_path):
    """
    Deletes 'text.txt', 'tables.txt', 'images_to_txt.txt' and any '.jpg' or '.png' files
    from the given folder after they have been concatenated into 'pdf.txt'.

    Parameters:
    - concat_file_path (str): Path to the concatenated 'pdf.txt' file.
    - folder_path (str): Path to the folder containing the files.
    """
    txt_files_to_delete = ["text.txt", "tables.txt", "images_to_txt.txt"]

    if os.path.exists(concat_file_path):
        # Delete specified text files
        for txt_filename in txt_files_to_delete:
            txt_path = os.path.join(folder_path, txt_filename)
            if os.path.exists(txt_path):
                try:
                    os.remove(txt_path)
                    logging.info(f"Deleted: {txt_path}")
                except Exception as e:
                    logging.error(f"Error deleting {txt_path}: {e}")

        # Delete any .jpg or .png files
        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, file)
                try:
                    os.remove(img_path)
                    logging.info(f"Deleted image: {img_path}")
                except Exception as e:
                    logging.error(f"Error deleting {img_path}: {e}")


def process_txt(folder_path):
    """
    Processes text files in labeled folders. For each folder containing
    ('text.txt', 'tables.txt', 'images_to_txt.txt'):

    1. Calls `concat_txt` to merge them into `pdf.txt`
    2. Calls `process_nlp` to clean up `pdf.txt`
    3. Calls `cleanup_txt_files` to delete intermediate files

    Parameters:
    - folder_path (str): Path to the root directory containing label folders.
    """

    for root, _, files in os.walk(folder_path):
        required_files = {"text.txt", "tables.txt", "images_to_txt.txt"}

        if required_files.intersection(set(files)):
            logging.info(f"Processing folder: {root}")

            try:
                concat_file_path = concat_txt(root, required_files)
                process_nlp(concat_file_path)
                cleanup_txt_files(concat_file_path,root)  # Cleanup files in the same directory
                logging.info(f"Successfully processed {root}")

            except Exception as e:
                logging.error(f"Error processing {root}: {e}")


def main(pdf_dir):
    if not os.path.isdir(pdf_dir):
        logging.error(f"{pdf_dir} is not a valid directory.")
        return

    for root, _, files in os.walk(pdf_dir):
        for filename in files:
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(root, filename)
                process_txt(root)
                logging.info(f"Processed: {pdf_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Process PDFs and extract text")
    parser.add_argument(
        '--pdf_path',
        type=str,
        default="../02-data/00-testing/coaxial-cables-rf",
        help="Path to the directory containing PDF files"
    )

    args = parser.parse_args()
    main(args.pdf_path)