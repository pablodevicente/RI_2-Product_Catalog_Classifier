import pandas as pd
import os
import logging
from tqdm import tqdm
from aux_extract_pdf import process_pdf
from transformers import AutoModelForCausalLM, AutoTokenizer

#base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'
#output_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs_txt/'
base_path = "../02-data/pdfs/"


logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_label(input_label_path):
    """
    Transcribes all PDFs within the input folder and saves the transcribed text
    in the corresponding output.

    Args:
        input_folder (str): The root folder containing labeled PDF files. # process_label("../02-data/pdfs/circuit-breakers")
    Raises:
        OSError: If there is an issue reading or writing files or directories.
    """
    try:
        # List directories (labels) within the input folder path
        pdf_labels = [f for f in os.listdir(input_label_path) if os.path.isdir(os.path.join(input_label_path, f))]
        for pdf_file_name in pdf_labels:
            # Correctly construct the full path including the label
            pdf_folder_path = os.path.join(input_label_path, pdf_file_name) ## ../02-data/pdfs/circuit-breakers/D_1110_en
            if os.path.isdir(pdf_folder_path):
                # Look for .pdf files and construct their full paths
                pdf_path = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.lower().endswith(".pdf")][0] ##only one element per folder, get the .pdf
                process_pdf(pdf_path,**model_kwargs)
            else:
                print(f"Invalid directory: {pdf_folder_path}")
    except OSError as e:
        logging.debug(f"File system error: {str(e)}")
    except Exception as e:
        logging.debug(f"Unexpected error during transcription: {str(e)}")

def check_folders(input_folder):
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
            input_path = os.path.join(input_folder, label)
            if os.path.isdir(input_path):
                process_label(input_path)  # process_label("../02-data/pdfs/circuit-breakers")

    except OSError as e:
        logging.error(f"File system error during label processing: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during label processing: {str(e)}")

def load_model(table_gpt):
    model_name = table_gpt
    # Another configuration at x10 parameters "tablegpt/TableGPT2-72B"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    example_prompt_template = """Given access to a pandas dataframes, answer the user question.

    /*
    "{var_name}.head(5).to_string(index=False)" as follows:
    {df_info}
    */

    Question: {user_question}
    """
    question = "Provide a detail description for each of the rows"

    # Return all arguments as a dictionary
    return {
        "model": model,
        "tokenizer": tokenizer,
        "prompt_template": example_prompt_template,
        "question": question,
    }

table_gpt = "tablegpt/TableGPT2-7B"
model_kwargs = load_model(table_gpt)

check_folders(base_path)
#process_folder(base_path, output_path)