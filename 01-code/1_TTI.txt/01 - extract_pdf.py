import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_extract_pdf import process_pdf,preload_model
import argparse
import traceback
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def check_folders(input_folder, **kwargs):
    """
    Performs checks and calls the 'process_pdf_file' function to process PDF files in the directory and its subdirectories.

    Args:
        input_folder (str): The root directory containing the sub-folders of PDF files.
            Example: '/path/to/root/01-pdfs/'
        kwargs (dict): Dictionary of keyword args with model information

    Returns:
        None: Logs results of the processing or errors encountered.
    """
    try:
        logging.info(f"Starting to process the input folder: {input_folder}")

        # Check if the folder exists before processing
        if not os.path.exists(input_folder):
            logging.error(f"Input folder does not exist: {input_folder}")
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        # Iterate through each subfolder and file recursively
        for root, dirs, files in os.walk(input_folder):
            logging.info(f"Currently processing directory: {root}")  # Log each directory

            for file in files:
                # Only process PDF files
                if file.lower().endswith(".pdf"):
                    pdf_path = os.path.join(root, file)
                    logging.info(f"Found PDF file: {pdf_path}. Starting processing.")

                    # Process the PDF file
                    table_processing = False
                    logging.info(
                        f"Table processing value is set to {table_processing} ------------------------- Beware")

                    try:
                        process_pdf(pdf_path, table_processing, **kwargs)
                    except Exception as e:
                        logging.error(f"Error processing PDF file {pdf_path}: {str(e)}")
                        logging.error(traceback.format_exc())  # Print full traceback for debugging

        logging.info(f"Completed processing all PDF files in folder: {input_folder}")

    except FileNotFoundError as e:
        logging.error(f"File not found error: {str(e)}")
        logging.error(traceback.format_exc())

    except OSError as e:
        logging.error(f"File system error during label processing: {str(e)}")
        logging.error(traceback.format_exc())

    except Exception as e:
        logging.error(f"Unexpected error during label processing: {str(e)}")
        logging.error(traceback.format_exc())


def main(base_path):
    """
    Main function to extract PDF data and process it using preloaded models.

    Args:
        base_path (str): The base path to the PDF folders.
    """
    # Preload models
    table_gpt = config.TABLE_GPT
    qwn = config.QWN
    model_kwargs = preload_model(model_gpt=table_gpt, model_qwn=qwn)

    # Process folders
    check_folders(base_path, **model_kwargs)


if __name__ == "__main__":
    # Call the main function with the provided base path
    main(config.BASE_PATH)