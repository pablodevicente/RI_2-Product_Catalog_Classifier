"""
Using the 222222github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
import os
import logging
import text_preprocessing as txtp
import re
import argparse
import aux_text_preprocessing as aux

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
            try:
                concat_file_path = aux.concat_txt(root, required_files)
                aux.process_nlp(concat_file_path)
                # cleanup_txt_files(concat_file_path,root)  # Cleanup files in the same directory
                logging.debug(f"Successfully processed {root}")

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

    parser = argparse.ArgumentParser(description="Process text files into a single file")
    parser.add_argument(
        '--pdf_path',
        type=str,
        default="../02-data/00-testing/",
        help="Path to the directory containing text files"
    )

    args = parser.parse_args()
    main(args.pdf_path)