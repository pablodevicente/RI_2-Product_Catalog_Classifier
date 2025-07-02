from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder, check_wifi_connection, create_urls
import pandas as pd
import logging
import os
import config
from pathlib import Path

base_path = config.BASE_PATH

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

        # Download the PDF and save it
        logger.debug(f"Downloading PDF from URL: {urls} to {save_folder}")
        download_pdfs_from_page(urls, save_fogit reset --soft HEAD~1
lder)

    logger.debug("Cleaning corrupted PDFs process initiated.")

    for label in categories.keys():
        logger.info(f"Processing corrupted PDFs for label: {label}.")
        save_folder = os.path.join(base_path, label)  # Folder to check for corrupted PDFs
        check_pdfs_in_folder(save_folder)         # Check for corrupted PDFs and handle them



def main():
    url_path = config.URL_PATH

    # Open the file and read the URLs line by line
    with open(url_path, "r") as file:
        urls = [line.strip() for line in file.readlines() if line.strip()]  # Remove extra spaces and newlines

    categories = create_urls(urls)

    # Call the main function and get the processed DataFrame
    download(categories)

if __name__ == "__main__":
    main()
