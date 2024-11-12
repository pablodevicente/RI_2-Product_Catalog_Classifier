import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm  # Correct import
import fitz  # PyMuPDF
import socket
import re
import logging
import hashlib
import json
import magic
import pandas as pd

# Configure the logger
logging.basicConfig(level=logging.ERROR, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_FILE = 'pdf_cache.json'

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def hash_content(content):
    return hashlib.md5(content).hexdigest()

def download_pdfs_from_page(urls, save_folder):
    """
    Downloads all PDFs from the provided URLs and saves them to the specified folder.
    Uses caching to avoid redundant downloads. Handles various infos (HTTP, connection, timeouts) 
    and logs events using the logger.
    
    Args:
        urls (list): A list of URLs to scrape for PDFs.
        save_folder (str): The folder path where the PDFs will be saved.
        
    Returns:
        None
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        logger.debug(f"Created directory: {save_folder}")

    cache = load_cache()

    for url in urls:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            logger.debug(f"Successfully retrieved webpage: {url}")
        except Exception as e:
            logger.debug(f"info retrieving webpage: {url}. Exception: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True) if '.pdf' in link['href']]

        if not pdf_links:
            logger.debug(f"No PDFs found on {url}.")
            continue

        total_processed = len(pdf_links)
        with tqdm(total=total_processed, desc=f"Downloading PDFs from {url}", unit="pdf") as pbar:
            for pdf_link in pdf_links:
                try:
                    pdf_response = requests.get(pdf_link, timeout=10)
                    if pdf_response.status_code == 200:
                        pdf_content_hash = hash_content(pdf_response.content)

                        if cache.get(pdf_link) == pdf_content_hash:
                            logger.debug(f"PDF {pdf_link} is already up-to-date. Skipping download.")
                        else:
                            pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                            with open(pdf_filename, 'wb') as f:
                                f.write(pdf_response.content)
                            cache[pdf_link] = pdf_content_hash
                            logger.debug(f"Successfully downloaded: {pdf_filename}")
                    else:
                        logger.debug(f"Failed to download {pdf_link}. Status code: {pdf_response.status_code}")
                except Exception as e:
                    logger.debug(f"info downloading PDF {pdf_link}: {e}")
                pbar.update(1)

        save_cache(cache)

def is_pdf_valid(file_path):
    """
    Checks if a given PDF file is valid by trying to open it and checking for pages.
    Additionally, it checks the actual file type to ensure it's a PDF.

    Args:
        file_path (str): The path to the PDF file to validate.
    
    Returns:
        bool: True if the PDF is valid, False otherwise.
    """
    try:
        # Use magic to check the actual file type
        mime = magic.from_file(file_path, mime=True)
        if mime != 'application/pdf':
            logger.debug(f"File '{file_path}' is not a valid PDF, it is '{mime}'")
            return False
        
        # Attempt to open the PDF and check if it has pages
        doc = fitz.open(file_path)  # Attempt to open the PDF
        if doc.page_count > 0:  # Valid PDF if it has at least one page
            doc.close()  # Close the file
            return True
    except Exception as e:
        logger.debug(f"Invalid PDF '{file_path}': {e}")
        return False
    return False

def check_pdfs_in_folder(folder_path):
    """
    Checks all PDF files in the given folder to determine if they are valid or invalid.
    Logs the results and deletes invalid files.

    Args:
        folder_path (str): The path to the folder containing PDF files.
    
    Returns:
        None
    """

   # Traverse the folder to check for PDF files
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Ignore hidden/system files
        if filename.startswith('.'):
            logger.debug(f"Skipping hidden/system file: {file_path}")
            continue

        if filename.lower().endswith('.pdf'):  # Only process PDF files
            # Check if the PDF is valid or invalid
            if is_pdf_valid(file_path):
                logger.debug(f"Valid PDF: {file_path}")
            else:
                logger.debug(f"Invalid PDF: {file_path} - Deleting file")
                if os.path.exists(file_path):
                    os.remove(file_path)  # Delete the invalid PDF
                else:
                    logger.debug(f"File not found: {file_path}")
        else:
            # Delete non-PDF files
            logger.debug(f"Non-PDF file: {file_path} - Deleting file")
            if os.path.exists(file_path):
                os.remove(file_path)  # Delete the non-PDF file
            else:
                logger.debug(f"File not found: {file_path}")


def check_wifi_connection():
    """
    Checks if the system has an active internet connection by trying to connect to a public DNS server.
    
    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        # Attempt to connect to Google's public DNS server (8.8.8.8) on port 53
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.debug("Internet connection is active.")
        return True
    except OSError:
        logger.debug("No internet connection detected.")
        return False

def create_urls(urls):
    """
    Given a list of base URLs, generates additional URLs by appending query parameters and organizes them
    into a catalog based on their subcategory (label) extracted from the URL.
    
    Args:
        urls (list): List of base URLs to process.
    
    Returns:
        dict: A catalog dictionary where the keys are subcategories (labels) and the values are lists of URLs.
    """
    
    def create_urls_and_append_to_catalog(base_url, catalog):
        """
        Helper function that generates additional URLs by appending specific query parameters to the base URL.
        It organizes the URLs into the provided catalog based on their subcategory.
        
        Args:
            base_url (str): The base URL to which query parameters will be appended.
            catalog (dict): Dictionary where URLs will be organized by their subcategory (label).
        
        Returns:
            None
        """
        # Extract the label from the base URL, assumed to be the second-to-last segment
        segments = base_url.split("/")
        label = segments[-2]  # Example: "https://example.com/filter/<label>/something"

        # List of additional query parameters to append to the base URL
        additional_queries = [
            "?s=N4IgrCBcoA5QjAGhDOl4AYMF9tA",
            "?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5WAaEM6QIwAYMF9tA",
            "?s=N4IgrCBcoA5QbAGhDOkCMAGTBfHQ"
        ]
        
        # Generate a list of full URLs (including the base URL)
        new_urls = [base_url + query for query in additional_queries]

        # Log the process of creating new URLs
        logger.debug(f"Generated URLs for label '{label}': {new_urls}")

        # Append the URLs to the catalog under the correct label
        if label in catalog:
            catalog[label].extend(new_urls)  # Append to existing list if label exists
        else:
            catalog[label] = new_urls  # Create a new entry if label doesn't exist
    
    # Initialize an empty catalog dictionary
    catalog = {}

    # Iterate over the provided URLs and process each one
    for url in urls:
        create_urls_and_append_to_catalog(url, catalog)

    # Log the final catalog structure
    logger.debug(f"Final URL catalog: {catalog}")
    
    return catalog


def count_files_in_folders(base_path, extension):
    """
    Count the number of files with a specific extension in each folder.

    Args:
        base_path (str): The directory to scan.
        extension (str): The file extension to count (e.g., ".pdf" or ".txt").

    Returns:
        dict: A dictionary where the keys are folder names and the values are the file counts.
    """
    folder_file_count = {}

    # Walk through the directory
    for root, dirs, files in os.walk(base_path):
        for folder in dirs:
            # Get the path of the folder
            folder_path = os.path.join(root, folder)

            # Count the number of files with the specified extension in the folder
            num_files = len([file for file in os.listdir(folder_path)
                             if os.path.isfile(os.path.join(folder_path, file)) and file.endswith(extension)])

            # Store the folder name and file count in the dictionary
            folder_file_count[folder] = num_files

    return folder_file_count


def generate_file_count_dataframe(pdf_path, txt_path, output_csv_path=None):
    """
    Generates a DataFrame with counts of PDF and text files in folders.

    Args:
        pdf_path (str): Path to the directory containing PDF folders.
        txt_path (str): Path to the directory containing text file folders.
        output_csv_path (str, optional): Path to save the CSV file. If None, the CSV is not saved.

    Returns:
        pd.DataFrame: A DataFrame with folder names, PDF counts, and text file counts.
    """
    # Count PDF and text files in respective folders
    pdfs_counts = count_files_in_folders(pdf_path, ".pdf")
    txt_counts = count_files_in_folders(txt_path, ".txt")

    # Create a pandas DataFrame from the PDF counts dictionary
    df = pd.DataFrame(list(pdfs_counts.items()), columns=['Folder', 'Pdf Count'])

    # Add Txt Count by matching the folder names from txt_counts
    df['Txt Count'] = df['Folder'].map(txt_counts)

    # Sort the DataFrame by 'Pdf Count' in descending order
    df_sorted = df.sort_values(by='Pdf Count', ascending=False)

    # Save the sorted DataFrame to a CSV file if an output path is specified
    if output_csv_path:
        df_sorted.to_csv(output_csv_path, index=False)

    return df_sorted


def find_missing_files(pdf_path, txt_path):
    """
    Finds files that don't appear in both pdf_path and txt_path directories.
    Compares the file names without extensions.

    Args:
        pdf_path (str): Directory containing PDF files.
        txt_path (str): Directory containing TXT files.

    Returns:
        missing_in_txt (list): List of PDFs that don't have a corresponding TXT file.
        missing_in_pdf (list): List of TXTs that don't have a corresponding PDF file.
    """
    # Get the list of PDF files and remove their extensions
    pdf_files = {os.path.splitext(f)[0] for f in os.listdir(pdf_path) if f.endswith(".pdf")}
    # Get the list of TXT files and remove their extensions
    txt_files = {os.path.splitext(f)[0] for f in os.listdir(txt_path) if f.endswith(".txt")}

    # Find PDFs without corresponding TXT files and vice versa
    missing_in_txt = pdf_files - txt_files  # PDFs that don't have TXT counterparts
    missing_in_pdf = txt_files - pdf_files  # TXTs that don't have PDF counterparts

    return missing_in_txt, missing_in_pdf