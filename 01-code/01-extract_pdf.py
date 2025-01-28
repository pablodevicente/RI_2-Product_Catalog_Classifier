import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_extract_pdf import process_pdf,preload_model
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_label(input_label_path, **kwargs):
    """
    Transcribes all PDFs within the input folder and saves the transcribed text in the corresponding output.

    Args:
        input_label_path (str): The root folder containing a label. Example: "../02-data/01-pdfs/circuit-breakers"
        kwargs (dict): Dictionary of keyword args with model information.

    Raises:
        OSError: If there is an issue reading or writing files or directories.
    """
    try:
        logging.debug(f"Starting to process label folder: {input_label_path}")

        # List all subfolders (PDF labels) within the label folder
        pdf_labels = [f for f in os.listdir(input_label_path) if os.path.isdir(os.path.join(input_label_path, f))]
        logging.debug(f"Found {len(pdf_labels)} subfolders in label folder: {input_label_path}")

        for pdf_file_name in pdf_labels:
            pdf_folder_path = os.path.join(input_label_path, pdf_file_name)  # ../02-data/01-pdfs/circuit-breakers/D_1110_en
            logging.debug(f"Processing subfolder: {pdf_folder_path}")

            if os.path.isdir(pdf_folder_path):
                # Find the PDF file in the current subfolder
                pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")]
                if pdf_files:
                    pdf_path = os.path.join(pdf_folder_path, pdf_files[0])  # ../02-data/01-pdfs/circuit-breakers/D_1110_en/D_1110_en.pdf
                    logging.debug(f"Found PDF file: {pdf_path}. Starting processing.")

                    # Process the PDF
                    process_pdf(pdf_path, **kwargs)
                    logging.debug(f"Completed processing of PDF: {pdf_path}")
                else:
                    logging.debug(f"No PDF files found in subfolder: {pdf_folder_path}")
            else:
                logging.debug(f"Skipping invalid or non-directory subfolder: {pdf_folder_path}")

        logging.debug(f"Completed processing all subfolders in label folder: {input_label_path}")

    except OSError as e:
        logging.error(f"File system error while processing label folder {input_label_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during processing of label folder {input_label_path}: {str(e)}")

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
        logging.debug(f"Starting to process the input folder: {input_folder}")

        # Iterate through each subfolder (label)
        for label in os.listdir(input_folder):
            input_path = os.path.join(input_folder, label)

            if os.path.isdir(input_path):
                logging.debug(f"Processing label: {label} (Path: {input_path})")
                process_label(input_path, **kwargs)  # Call process_label for the label
            else:
                logging.debug(f"Skipping non-directory item: {label} (Path: {input_path})")

        logging.debug(f"Completed processing all labels in folder: {input_folder}")

    except OSError as e:
        logging.error(f"File system error during label processing: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during label processing: {str(e)}")


def main(base_path="../02-data/01-pdfs/"):
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
        help="The base path to the folder containing PDF files."
    )
    args = parser.parse_args()

    # Call the main function with the provided base path
    main(args.base_path)