import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_pdf_to_text import process_folder

#base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'
#output_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs_txt/'
base_path = "02-data/pdfs/multiple-conductor-cables/"
output_path = "02-data/txts/multiple-conductor-cables/"

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compare_folders(input_folder, output_folder):
    """
    Compares the number of documents in each corresponding folder within
    the input and output directories. If the number of documents doesn't match,
    calls the 'save_transcribed_pdfs' function to process the label.

    Args:
        input_folder (str): The root directory containing the original PDF files. # /pdfs/
        output_folder (str): The directory containing the transcribed text files. # /pdfs_txt/
    
    Returns:
        None: Logs results of the comparison.
    """
    try:
        for label in os.listdir(input_folder):
            input_path = os.path.join(input_folder, label)    # /pdfs/label1/
            output_path = os.path.join(output_folder, label)  # /txts/label1/

            if os.path.isdir(input_path):
                # if output does not exist, create it
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    logging.debug(f"Created output folder: {output_path}")

                # Count PDF and TXT files
                pdfs_count = len([f for f in os.listdir(input_path) if f.endswith(".pdf")])
                pdfs_txt_count = len([f for f in os.listdir(output_path) if f.endswith(".txt")])

                # Compare counts, and process if not matching
                if pdfs_count != pdfs_txt_count: # dosent make sense if each image is its own separate file
                    logging.info(f"Processing '{label}'")
                    process_folder(input_path, output_path)
                else:
                    logging.info(f"Folder '{label}': PDF and TXT file counts match ({pdfs_count}). (not really, check condition)")
            else:
                logging.warning(f"'{input_path}' is not a valid folder. Skipping...")

    except OSError as e:
        logging.error(f"File system error during folder comparison: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during folder comparison: {str(e)}")


#compare_folders(base_path, output_path)
process_folder(base_path, output_path)