from aux_download_pdfs import download_pdfs_from_page, check_pdfs_in_folder, check_wifi_connection, create_urls
import pandas as pd
import logging
import os

# Define the variable for the first part of the path
base_path = '/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/'

# Configure the logger
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(categories, debug=False):
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
        
        # Download PDFs from the URLs
        download_pdfs_from_page(urls, save_folder)

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
    main(categories)