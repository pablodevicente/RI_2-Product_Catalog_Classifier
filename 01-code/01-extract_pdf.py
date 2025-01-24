import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_extract_pdf import process_pdf,preload_model

base_path = "../02-data/01-pdfs/"

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_label(input_label_path,kwargs):
    """
    Transcribes all PDFs within the input folder and saves the transcribed text in the corresponding output.

    Args:
        input_label_path (str): The root folder containing a label. # process_label("../02-data/01-pdfs/circuit-breakers")
        kwargs (dict): Dictionary of keyword args with model information

    Raises:
        OSError: If there is an issue reading or writing files or directories.
    """
    try:
        # Lists pdfs within a label.
        pdf_labels = [f for f in os.listdir(input_label_path) if os.path.isdir(os.path.join(input_label_path, f))]
        for pdf_file_name in pdf_labels:
            pdf_folder_path = os.path.join(input_label_path, pdf_file_name) ## ../02-data/01-pdfs/circuit-breakers/D_1110_en

            if os.path.isdir(pdf_folder_path):
                # Look for .pdf files and construct their full paths ## ../02-data/01-pdfs/circuit-breakers/D_1110_en/D_1110_en
                pdf_path = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")][0] ##only one element per folder, get the .pdf

                process_pdf(pdf_path,**kwargs)

            else:
                logging.debug(f"Invalid directory: {pdf_folder_path}")
    except OSError as e:
        logging.debug(f"File system error: {str(e)}")
    except Exception as e:
        logging.debug(f"Unexpected error during transcription: {str(e)}")

def check_folders(input_folder,**kwargs):
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
        for label in os.listdir(input_folder):
            input_path = os.path.join(input_folder, label)
            if os.path.isdir(input_path):
                process_label(input_path,**kwargs)  # process_label("../02-data/01-pdfs/circuit-breakers")

    except OSError as e:
        logging.error(f"File system error during label processing: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during label processing: {str(e)}")


table_gpt = "tablegpt/TableGPT2-7B"
qwn = "Qwen/Qwen2.5-7B-Instruct"

model_kwargs = preload_model(model_gpt=table_gpt,model_qwn=qwn)

check_folders(base_path,**model_kwargs)
