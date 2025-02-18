import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_extract_pdf import process_pdf,preload_model,process_label
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

def check_folders(input_folder, **kwargs):
    """
    Performs some checks and calls the 'process_label' function to process the label.

    Args:
        input_folder (str): The root directory containing the sub-folders of original PDF files.
            Example: '/path/to/root/01-pdfs/'
        kwargs (dict): Dictionary of keyword args with model information

    Returns:
        None: Logs results of the comparison or errors encountered.
    """
    try:
        logging.info(f"Starting to process the input folder: {input_folder}")

        # Iterate through each subfolder (label)
        for label in os.listdir(input_folder):
            input_path = os.path.join(input_folder, label)

            if os.path.isdir(input_path):
                logging.info(f"Processing label: {label} (Path: {input_path})")
                process_label(input_path, **kwargs)  # Call process_label for the label
            else:
                logging.info(f"Skipping non-directory item: {label} (Path: {input_path})")

        logging.info(f"Completed processing all labels in folder: {input_folder}")

    except OSError as e:
        logging.error(f"File system error during label processing: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during label processing: {str(e)}")


def main(base_path):
    """
    Main function to extract PDF data and process it using preloaded models.

    Args:
        base_path (str): The base path to the PDF folders.
    """
    # Preload models
    table_gpt = "tablegpt/TableGPT2-7B"
    qwn = "Qwen/Qwen2.5-7B-Instruct"
    model_kwargs = preload_model(model_gpt=table_gpt, model_qwn=qwn)

    # Process folders
    check_folders(base_path, **model_kwargs)


if __name__ == "__main__":
    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Extract and process PDF data.")
    parser.add_argument(
        "base_path",
        type=str,
        nargs="?",  # Makes the argument optional
        default="../02-data/00-testing/03-demo/",
        help="The base path to the folder containing PDF files."
    )
    args = parser.parse_args()

    # Call the main function with the provided base path
    main(args.base_path)