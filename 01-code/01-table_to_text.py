import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
import tabulate
import os
import logging

base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'
output_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs_txt/'

# Configure the logger
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def process_pdf(pdf_path):
    """
    Processes a single PDF file to extract its text and tables.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text and tables in markdown format.
    """
    try:
        pdf = pdfplumber.open(pdf_path)
        all_text = []
        tables_read = 0
        tables_missed = 0

        # Iterate through each page of the PDF
        for page in pdf.pages:
            # Extract all text from the page before filtering tables
            page_text = page.extract_text()

            chars = page.chars  # Get the characters on the page
            for table in page.find_tables():
                try:
                    # Try to find the first character in the cropped table
                    first_table_char = page.crop(table.bbox).chars[0]
                    tables_read += 1
                except IndexError:
                    tables_missed += 1
                    continue  # If the table is empty, skip to the next one

                # Filter out the table's characters from the page
                filtered_page = page.filter(lambda obj: 
                    get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
                )
                chars = filtered_page.chars

                # Convert the table to a DataFrame and then to markdown format
                df = pd.DataFrame(table.extract())
                df.columns = df.iloc[0]  # Use the first row as header
                markdown = df.drop(0).to_markdown(index=False)

                # Append the markdown representation of the table to the characters
                chars.append(first_table_char | {"text": markdown})

            # If there is page text, append it first
            if page_text:
                all_text.append(page_text)

            # Add the text after processing the tables
            page_text_with_tables = extract_text(chars, layout=True)
            all_text.append(page_text_with_tables)

        pdf.close()
        logging.debug(f"Successfully read {tables_read} tables and missed {tables_missed} tables from {pdf_path}.")
        return "\n".join(all_text)

    except Exception as e:
        logging.debug(f"Error processing PDF {pdf_path}: {str(e)}")
        return ""


def save_transcribed_pdfs(input_folder, output_folder):
    """
    Transcribes all PDFs within the input folder and saves the transcribed text
    in the corresponding output folder with the same folder and file structure.

    Args:
        input_folder (str): The root folder containing labeled PDF files.
        output_folder (str): The destination folder for the transcribed text files.

    Raises:
        OSError: If there is an issue reading or writing files or directories.
    """
    try:
        # Loop through each label folder in the input directory (e.g., pdfs/)
        for label in os.listdir(input_folder):
            label_path = os.path.join(input_folder, label)

            # Only process folders (labels)
            if os.path.isdir(label_path):
                # Create corresponding label folder in the output directory (e.g., pdfs_txt/label/)
                label_output_folder = os.path.join(output_folder, label)
                os.makedirs(label_output_folder, exist_ok=True)

                # Loop through each PDF in the label folder
                for pdf_file in os.listdir(label_path):
                    if pdf_file.endswith(".pdf"):
                        pdf_path = os.path.join(label_path, pdf_file)

                        # Process the PDF and get the transcribed text
                        transcribed_text = process_pdf(pdf_path)

                        if transcribed_text:
                            # Save the transcribed text to a .txt file with the same name as the PDF
                            document_name = os.path.splitext(pdf_file)[0]  # Remove the file extension
                            output_txt_path = os.path.join(label_output_folder, f"{document_name}.txt")
                            
                            # Write transcribed text to file
                            with open(output_txt_path, "w", encoding="utf-8") as text_file:
                                text_file.write(transcribed_text)

                            logging.debug(f"Transcribed PDF saved at: {output_txt_path}")
                        else:
                            logging.debug(f"No text extracted from {pdf_path}")
            else:
                logging.debug(f"{label_path} is not a folder, skipping...")

    except OSError as e:
        logging.debug(f"File system error: {str(e)}")
    except Exception as e:
        logging.debug(f"Unexpected error during transcription: {str(e)}")


# Example usage
save_transcribed_pdfs(base_path, output_path)