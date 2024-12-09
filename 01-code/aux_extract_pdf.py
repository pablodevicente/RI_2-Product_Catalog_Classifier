import pandas as pd
import os
import logging
from tqdm import tqdm
import fitz
import pdfplumber


base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'
output_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs_txt/'

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf):
    """
    Extracts text from each page in the PDF.
    """
    all_text = []
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            all_text.append(page_text)
    return "\n".join(all_text)

"""
Functions related to table extraction, preprocessing, categorizacion and table_to_text
Done at the same time as the processing of the pdf because of reasons, but could be done in a different step 
(convert df to markdown, write onto file and the markdown to df for easier handling didnt seem efficient)
"""

def extract_tables_from_pdf(pdf,**kwargs):
    """
    Extracts tables from each page in the PDF and converts them to markdown format.
    """
    all_tables = []

    for page in pdf.pages:
        for table in page.find_tables():

            # Step 1: Extract the dataframe into pandas df format
            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]  # Use the first row as header
            # Step 2: Clean the dataframe
            df_clean = clean_table(df)

            # Check if the DataFrame is empty
            if df_clean.empty:
                # logger.DEBUG("DataFrame is empty. Skipping further processing.")
                pass
            else:

                # Step 3: Filter the dataframe into one of two types
                typing = categorize_dataframe(df_clean)

                if typing == 0:                    # call table_gpt

                    prompt = kwargs["example_prompt_template"].format(
                        var_name="df",
                        df_info=df.head(5).to_string(index=False),
                        user_question=kwargs["question"],
                    )

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                    text = kwargs["tokenizer"].apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    model_inputs = kwargs["tokenizer"]([text], return_tensors="pt").to(kwargs["model"].device)

                    generated_ids = kwargs["model"].generate(**model_inputs, max_new_tokens=512)
                    generated_ids = [
                        output_ids[len(input_ids):]
                        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                    response = kwargs["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]

                    all_tables.append(response)

                else:
                    print("----------1")
                    pass
                    # call llm to describe single row

    return "\n".join(all_tables)


def clean_table(df):
    """
    Cleans a table (DataFrame) by applying various transformations:
    - Removes horizontal tables. -- not implemented
    - Removes duplicate columns (case-insensitive).
    - Removes columns containing only underscores.
    - Removes columns with field content exceeding 200 characters. -- originally 100, but i may get some more
    - Removes rows/columns with more than 30% NaN values. --  mnnn not really taking out rows, didnt work well
    - Removes columns where the first value matches the column name.

    Parameters:
    - df: Pandas DataFrame representing the table.

    Returns:
    - A cleaned DataFrame.
    """
    # Step 1: Remove horizontal tables
    # if len(df.columns) > len(df) * 2:  # Assuming horizontal if columns are more than twice the rows
    #    return None  # Skip horizontal tables

    # Step 2: Remove duplicate columns (case-insensitive)
    df = df.loc[:, ~df.columns.str.lower().duplicated()]

    # Step 3: Remove columns containing only underscores
    df = df.loc[:, ~(df.apply(lambda col: col.str.fullmatch(r"_+").all(), axis=0))]

    # Step 4: Remove columns where field content exceeds 200 characters
    df = df.loc[:, ~(df.apply(lambda col: col.astype(str).map(len).max() > 200, axis=0))]

    # Step 5.0 : Replace empty strings for NaN values
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    # Step 5: Remove rows/columns with more than 30% NaN values
    threshold = 0.3

        # Step 5.1: Drop columns with more than 30% NaN values
    col_thresh = int((1 - threshold) * len(df))  # Minimum non-NaN values required
    df = df.dropna(axis=1, thresh=col_thresh)

        # Step 5.2: Drop rows with more than 30% NaN values
    # didnt work to well

        # Step 5.3: if one column. Delete NaNs
    if df.shape[1] == 1:
        df = df.dropna(axis=0)

    # Check if the DataFrame is empty after Step 5
    if not df.empty:
        # Step 6: Remove columns where the first value matches the column name
        matches = (df.iloc[0].str.lower() == df.columns.str.lower())

        # Count how many values in the first row match the column names
        match_count = matches.sum()

        # If more than 30% of the values match, remove the first row
        if match_count / len(df.columns) > 0.3:
            df = df.drop(index=0)

    return df.reset_index(drop=True)  # Reset index for clean DataFrame

def categorize_dataframe(df):
    """
    Categorizes a DataFrame based on its dimensions. (Made a function for easier alteration)

    Parameters:
        df (pd.DataFrame): The DataFrame to categorize.

    Returns:
        int: 0 if the DataFrame has more than 5 rows or more than 2 columns (both included),
             1 otherwise.
    """
    if df.shape[0] > 5 and df.shape[1] > 2:
        return 0
    else:
        return 1

"""
Image extraction functions, not using all of them, some dead code
"""
def extract_images_from_pdf_fitz(pdf_path, pdf_name, output_folder):
    """
    Extracts images from each page in the PDF and saves them in the specified folder.

    Args:
        pdf_path (str): Path to the PDF file.
        pdf_name (str): Base name for the output files.
        output_folder (str): Folder where extracted images will be saved.
    """
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

    pdf = fitz.open(pdf_path)
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            img_data = base_image["image"]
            img_ext = base_image["ext"]

            # Define the output path and save the image data
            img_path = os.path.join(output_folder, f"{pdf_name}_page{page_num + 1}_img{img_index + 1}.{img_ext}")
            with open(img_path, "wb") as img_file:
                img_file.write(img_data)
            logger.debug(f"Saved image to {img_path}")
    logger.debug(f"Images processed from file {pdf_name}")
    pdf.close()

"""
Main function, just calls all other methods
"""

def process_pdf(pdf_name, pdf_path, output_folder,**model_kwargs):
    """
    Processes a single PDF file to extract text, tables, and images.
    Extracts the text and tables and images to individual files
    Uses fitz for image selection

    Args:
        pdf_name (str): Name of the PDF file without extension.
        pdf_path (str): The path to the PDF file.
        output_folder (str): Folder to save extracted data for each PDF.

    Returns:
        None
    """
    try:
        # Define the PDF-specific output folder
        pdf_output_folder = os.path.join(output_folder, pdf_name)
        os.makedirs(pdf_output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        # Open the PDF
        pdf = pdfplumber.open(pdf_path)

        # Step 1: Extract text and save to text.txt
        text_content = extract_text_from_pdf(pdf)
        text_output_path = os.path.join(pdf_output_folder, "text.txt")
        with open(text_output_path, "w") as text_file:
            text_file.write(text_content)

        # Step 2: Extract tables, translate onto text and save to text.txt
        tables_content = extract_tables_from_pdf(pdf,**model_kwargs)
        text_file.write(tables_content)

        # Step 3: Extract and save images
        extract_images_from_pdf_fitz(pdf_path, pdf_name, pdf_output_folder)

        pdf.close()
        logging.debug(f"Successfully processed {pdf_path}.")

    except Exception as e:
        logging.debug(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""
