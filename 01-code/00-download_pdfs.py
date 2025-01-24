from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder, check_wifi_connection, create_urls
import pandas as pd
import logging
import os

# Define the variable for the first part of the path
base_path = '/02-data/01-pdfs/'

# Configure the logger
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download(categories, debug=False):
    """
    Main function to download PDFs and optionally clean corrupted files.
    
    Args:
        categories (dict): Dictionary where keys are labels (categories) and values are lists of URLs.
        clean (bool): If True, will also check and clean corrupted PDFs after download.
    
    Returns:
        None
    """
    
    # Check Wi-Fi connection
    if check_wifi_connection():
        logger.debug("Wi-Fi is connected.")
    else:
        logger.debug("Wi-Fi is not connected. Exiting execution.")
        return  
    
    # Download PDFs for each category
    for label, urls in categories.items():
        save_folder = os.path.join(base_path, label)  # Create save path for each label
        logger.debug(f"Processing label: {label} with {len(urls)} URLs.")

        for url in urls:
            # Extract the PDF name from the URL
            pdf_name = os.path.splitext(os.path.basename(url))[0]

            # Create a subfolder for each PDF
            pdf_folder = os.path.join(save_folder, pdf_name)
            os.makedirs(pdf_folder, exist_ok=True)  # Ensure the folder exists

            # Define the full path to save the PDF
            pdf_path = os.path.join(pdf_folder, f"{pdf_name}.pdf")

            # Download the PDF and save it
            logger.debug(f"Downloading PDF from URL: {url} to {pdf_path}")
            download_pdfs_from_page(url, pdf_path)

    logger.debug("Cleaning corrupted PDFs process initiated.")
    for label in categories.keys():
        logger.info(f"Processing corrupted PDFs for label: {label}.")
        save_folder = os.path.join(base_path, label)  # Folder to check for corrupted PDFs
        check_pdfs_in_folder(save_folder)         # Check for corrupted PDFs and handle them


if __name__ == "__main__":

    # Open the file and read the URLs line by line
    with open("02-data/urls.txt", "r") as file:
        urls = [line.strip() for line in file.readlines() if line.strip()]  # Remove extra spaces and newlines

    categories = create_urls(urls)
    # Call the main function and get the processed DataFrame
    download(categories)