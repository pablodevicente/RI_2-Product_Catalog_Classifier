import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm  # Correct import
import fitz  # PyMuPDF
import socket
import re
import requests
import logging
from requests.exceptions import ConnectionError, Timeout

# Configure the logger
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_pdfs_from_page(urls, save_folder):
    """
    Downloads all PDFs from the provided URLs and saves them to the specified folder.
    Handles various errors (HTTP, connection, timeouts) and logs events using the logger.
    
    Args:
        urls (list): A list of URLs to scrape for PDFs.
        save_folder (str): The folder path where the PDFs will be saved.
        
    Returns:
        None
    """
    # Check if folder exists and has fewer than 10 files, if not, create the folder
    if not os.path.exists(save_folder) or len(os.listdir(save_folder)) <= 10:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            logger.info(f"Created directory: {save_folder}")
        else:
            logger.info(f"Directory already exists with less than 10 files: {save_folder}")
        
        for url in urls:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', 60)  # Default to 60 seconds if header is missing
                    logger.error(f"Rate limit exceeded for {url}. Retry after {retry_after} seconds.")
                    return  # Finish execution if rate limit is hit
                elif response.status_code != 200:
                    logger.error(f"Failed to retrieve webpage from {url}: {response.status_code}")
                    continue
                response.raise_for_status()  # Raise HTTPError for bad responses
                logger.info(f"Successfully retrieved webpage: {url}")
                
            except ConnectionError:
                logger.error(f"No route to host: {url}. Please check your network connection.")
                continue
            except requests.exceptions.HTTPError as err:
                logger.error(f"HTTP error occurred for {url}: {err}")
                continue
            except Exception as e:
                logger.error(f"An unexpected error occurred while accessing {url}: {e}")
                continue

            # Parse the webpage content to find PDF links
            soup = BeautifulSoup(response.content, 'html.parser')
            pdf_links = [urljoin(url, link['href']) for link in soup.find_all('a', href=True) if '.pdf' in link['href']]

            if not pdf_links:
                logger.info(f"No PDFs found on {url}.")
                continue

            # Statistics for downloads
            total_success, total_failed, total_exceptions = 0, 0, 0
            total_processed = len(pdf_links)

            with tqdm(total=total_processed, desc=f"Downloading PDFs from {url}", unit="pdf") as pbar:
                for pdf_link in pdf_links:
                    try:
                        pdf_response = requests.get(pdf_link, timeout=10)
                        if pdf_response.status_code == 200:
                            pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                            with open(pdf_filename, 'wb') as f:
                                f.write(pdf_response.content)
                            total_success += 1
                            logger.info(f"Successfully downloaded: {pdf_filename}")
                        else:
                            total_failed += 1
                            logger.info(f"Failed to download {pdf_link}. Status code: {pdf_response.status_code}")
                    except Timeout:
                        total_exceptions += 1
                        logger.info(f"Timeout occurred while downloading: {pdf_link}")
                    except Exception as e:
                        total_exceptions += 1
                        logger.info(f"An error occurred while downloading {pdf_link}: {e}")

                    pbar.update(1)  # Update progress bar even if an error occurs

            # Log final statistics for this URL
            logger.info(f"Completed processing {url}: Success: {total_success}, Failed: {total_failed}, Exceptions: {total_exceptions}")
    else:
        logger.info(f"Directory '{save_folder}' already contains more than 10 files. Skipping download.")

def is_pdf_valid(file_path):
    """
    Checks if a given PDF file is valid by trying to open it and checking for pages.

    Args:
        file_path (str): The path to the PDF file to validate.
    
    Returns:
        bool: True if the PDF is valid, False otherwise.
    """
    try:
        doc = fitz.open(file_path)  # Attempt to open the PDF
        if doc.page_count > 0:  # Valid PDF if it has at least one page
            doc.close()  # Close the file
            return True
    except Exception as e:
        logger.error(f"Invalid PDF '{file_path}': {e}")
        return False
    return False

def check_pdfs_in_folder(folder_path):
    """
    Checks all PDF files in the given folder to determine if they are valid or invalid.
    Logs the results, including percentages of valid and invalid files.

    Args:
        folder_path (str): The path to the folder containing PDF files.
    
    Returns:
        None
    """
    # Statistics
    total_files = 0
    valid_files = 0
    invalid_files = 0

    # Traverse the folder to check for PDF files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):  # Only process PDF files
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            # Check if the PDF is valid or invalid
            if is_pdf_valid(file_path):
                valid_files += 1
                logger.info(f"Valid PDF: {file_path}")
            else:
                invalid_files += 1
                logger.warning(f"Invalid PDF: {file_path}")

    try:
        # Calculate percentages
        valid_percentage = (valid_files / total_files) * 100 if total_files != 0 else 0
        invalid_percentage = (invalid_files / total_files) * 100 if total_files != 0 else 0

        # Log the final statistics
        logger.info(f"Total PDFs: {total_files} | "
                    f"Valid: {valid_files} ({valid_percentage:.2f}%) | "
                    f"Invalid: {invalid_files} ({invalid_percentage:.2f}%)")
    except ZeroDivisionError:
        logger.warning("No files processed. Division by zero occurred.")

def check_wifi_connection():
    """
    Checks if the system has an active internet connection by trying to connect to a public DNS server.
    
    Returns:
        bool: True if the connection is successful, False otherwise.
    """
    try:
        # Attempt to connect to Google's public DNS server (8.8.8.8) on port 53
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.info("Internet connection is active.")
        return True
    except OSError:
        logger.error("No internet connection detected.")
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
            "?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ]
        
        # Generate a list of full URLs (including the base URL)
        new_urls = [base_url] + [base_url + query for query in additional_queries]

        # Log the process of creating new URLs
        logger.info(f"Generated URLs for label '{label}': {new_urls}")

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
    logger.info(f"Final URL catalog: {catalog}")
    
    return catalog