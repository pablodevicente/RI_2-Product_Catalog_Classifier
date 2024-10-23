import PyPDF2
import pytesseract
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF

pdf_path = "02-data/original_examples/"
file_name = "pdf_file_4137"

file_path = pdf_path + file_name + ".pdf"

def pdf_to_text_pypdf2(pdf_path, txt_output_path):
    try:
        # Open the PDF file in read-binary mode
        with open(pdf_path, 'rb') as pdf_file:
            # Initialize the PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_content = ""

            # Iterate through all the pages and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()

            # Write the extracted text to a .txt file
            with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)
        
        print(f"Text successfully extracted and saved to {txt_output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

txt_output_path = pdf_path + file_name + "_pypdf2.txt"
pdf_to_text_pypdf2(file_path, txt_output_path)

def image_to_text_tesseract(image_path, output_txt_path):
    try:
        # Open the image file
        img = Image.open(image_path)

        # Use pytesseract to do OCR on the image
        # Enable layout analysis for multi-column layout
        extracted_text = pytesseract.image_to_string(img, config='--psm 6')

        # Save the extracted text to a file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(extracted_text)
        
        print(f"Text successfully extracted and saved to {output_txt_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

txt_output_path = pdf_path + file_name + "_pytesseract.txt"
image_to_text_tesseract(file_path, txt_output_path)

def pdf_to_text_with_layout(pdf_path, txt_output_path):
    try:
        # Open the PDF file
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""

            # Iterate through all the pages in the PDF
            for page_num, page in enumerate(pdf.pages):
                # Extract text while preserving layout
                text = page.extract_text()
                
                if text:
                    text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
                    text_content += text

            # Save the extracted text to a .txt file
            with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)

        print(f"Text successfully extracted and saved to {txt_output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

txt_output_path = pdf_path + file_name + "_pdfplumber.txt"
pdf_to_text_with_layout(file_path, txt_output_path)

def pdf_to_text_pymupdf(pdf_path, txt_output_path):
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        text_content = ""

        # Iterate through all the pages in the PDF
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # Extract text with PyMuPDF, which preserves layout better
            text = page.get_text("text")
            
            text_content += f"\n\n--- Page {page_num + 1} ---\n\n"
            text_content += text

        # Save the extracted text to a .txt file
        with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)

        print(f"Text successfully extracted and saved to {txt_output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

txt_output_path = pdf_path + file_name + "_pymupdf.txt"
pdf_to_text_pymupdf(file_path, txt_output_path)