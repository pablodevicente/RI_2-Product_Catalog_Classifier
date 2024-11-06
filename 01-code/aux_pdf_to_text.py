import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
import tabulate
import os
import logging
from tqdm import tqdm
from PIL import Image
import fitz

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

def extract_tables_from_pdf(pdf):
    """
    Extracts tables from each page in the PDF and converts them to markdown format.
    """
    all_tables = []
    for page in pdf.pages:
        for table in page.find_tables():
            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]  # Use the first row as header
            markdown = df.drop(0).to_markdown(index=False)
            all_tables.append(markdown)
    return "\n\n".join(all_tables)

def extract_images_from_pdf_pdfplumber(pdf, pdf_name, output_folder): ## deprecated. fitz works infinitelly better

    for page_num, page in enumerate(pdf.pages, start=1):
        for img_index, image in enumerate(page.images):
            # Extract image properties
            x0, y0, x1, y1 = image["x0"], image["y0"], image["x1"], image["y1"]
            # Crop the image region from the page
            cropped_image = page.within_bbox((x0, y0, x1, y1)).to_image()
            # Save as a file
            cropped_image.save(f"{output_folder}"f"{pdf_name}_page{page_num}_img{img_index + 1}.png", format="PNG")


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


def process_pdf(pdf_name, pdf_path, output_folder):
    """
    Processes a single PDF file to extract text, tables, and images.

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

        # Step 2: Extract tables and save to tables.txt
        tables_content = extract_tables_from_pdf(pdf)
        tables_output_path = os.path.join(pdf_output_folder, "tables.txt")
        with open(tables_output_path, "w") as tables_file:
            tables_file.write(tables_content)

        # Step 3: Extract and save images
        extract_images_from_pdf_fitz(pdf_path, pdf_name, pdf_output_folder)

        pdf.close()
        logging.debug(f"Successfully processed {pdf_path}.")

    except Exception as e:
        logging.debug(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""

def process_folder(input_folder_path, output_folder_path):
    """
    Transcribes all PDFs within the input folder and saves the transcribed text
    in the corresponding output folder with the same file structure.

    Args:
        input_folder (str): The root folder containing labeled PDF files.
        output_folder (str): The destination folder for the transcribed text files.

    Raises:
        OSError: If there is an issue reading or writing files or directories.
    """
    try:

        pdf_files = [f for f in os.listdir(input_folder_path) if f.endswith(".pdf")]

        for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):

            pdf_path = os.path.join(input_folder_path, pdf_file) # /02-data/pdfs/label_x/pdf_file_y.pdf
            pdf_name = os.path.splitext(pdf_file)[0]  # pdf_file_y

            process_pdf(pdf_name, pdf_path, output_folder_path)

            logging.debug(f"Transcribed PDF saved at: {output_folder_path}")


    except OSError as e:
        logging.debug(f"File system error: {str(e)}")
    except Exception as e:
        logging.debug(f"Unexpected error during transcription: {str(e)}")
