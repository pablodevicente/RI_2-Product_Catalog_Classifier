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
from requests.exceptions import ConnectionError


# Configure the logger
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_pdfs_from_page(urls, save_folder):
    
    if not os.path.exists(save_folder) or len(os.listdir(save_folder)) <= 10: # if the path exists but has les than x files then download
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for url in urls:

            # Send request to fetch content of the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            
            try:
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    logger.error(f"Failed to retrieve webpage: {response.status_code}")
                    return
                response.raise_for_status()  # Raise HTTPError for bad responses
            except ConnectionError:
                logger.error("No route to host. Please check your network connection.")
            except requests.exceptions.HTTPError as err:
                logger.error(f"HTTP error occurred: {err}")
            except Exception as e:
                logger.error(f"An error occurred: {e}")
            else:
                # Process the response if everything went well
                logger.info("Request was successful.")

            # Parse the webpage content
            soup = BeautifulSoup(response.content, 'html.parser')

            pdf_links = []
            # Find all <a> tags with href attributes that contain '.pdf'
            for link in soup.find_all('a', href=True):  # find all <a> tags with href attribute
                href = link['href']
                if '.pdf' in href:
                    # Handle relative URLs that start with //
                    if href.startswith('//'):
                        href = 'https:' + href
                    pdf_url = urljoin(url, href)
                    pdf_links.append(pdf_url)

            # Find PDFs embedded in JavaScript or JSON
            # This uses a regular expression to search for any '.pdf' in the entire HTML content -> does not find any additional pdf, all duplicates
            # potential_pdfs = re.findall(r'\"(https?://[^\"]+\.pdf)\"', response.text)
            # pdf_links.extend(potential_pdfs)

            # Statistics
            total_processed = len(pdf_links)
            total_success,total_failed,total_exceptions = 0,0,0
 
            with tqdm(total=total_processed, desc="Downloading PDFs from url", unit="pdf") as pbar:
                for pdf_link in pdf_links:
                    try:
                        pdf_response = requests.get(pdf_link, timeout=3)  # Set timeout to 10 seconds
                        if pdf_response.status_code == 200:
                            pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                            with open(pdf_filename, 'wb') as f:
                                f.write(pdf_response.content)
                            total_success += 1 
                        else:
                            total_failed += 1  
                    except requests.exceptions.Timeout:
                        #print(f"Timeout occurred while downloading: {pdf_link}")
                        total_exceptions += 1
                    except Exception as e:
                        #print(f"An error occurred while downloading: {pdf_link}, Error: {e}")
                        total_exceptions += 1  
                    
                    pbar.update(1)  # Update progress bar even if an error occurs


            try:
                success_percentage = (total_success / total_processed) * 100 if total_processed != 0 else 0
                failed_percentage = (total_failed / total_processed) * 100 if total_processed != 0 else 0
                exception_percentage = (total_exceptions / total_processed) * 100 if total_processed != 0 else 0

                print(f"Total: {total_processed} / 100 | "
                    f"Downloaded: {total_success} ({success_percentage:.2f}%) | "
                    f"Failed: {total_failed} ({failed_percentage:.2f}%) | "
                    f"Exceptions: {total_exceptions} ({exception_percentage:.2f}%)")
            except ZeroDivisionError:
                print("No PDFs processed")

def is_pdf_valid(file_path):
    try:
        doc = fitz.open(file_path)
        if doc.page_count > 0:
            doc.close()  
            return True
        
    except Exception as e:
        #print(f"Invalid pdf: {e}")
        return False
    
    return False  # Not a valid PDF if no pages are found


def check_pdfs_in_folder(folder_path):
    
    # Statistics
    total_files = 0
    valid_files = 0
    invalid_files = 0

    # Look for invalid files
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'): 
            total_files += 1
            file_path = os.path.join(folder_path, filename)

            if is_pdf_valid(file_path):
                valid_files += 1
            else:
                invalid_files += 1

    try:
        valid_percentage = (valid_files / total_files) * 100 if total_files != 0 else 0
        invalid_percentage = (invalid_files / total_files) * 100 if total_files != 0 else 0

        print(f"Total: {total_files} | "
            f"Valid: {valid_files} ({valid_percentage:.2f}%) | "
            f"Invalid: {invalid_files} ({invalid_percentage:.2f}%)")
    except ZeroDivisionError:
        print("No files processed yet, so division by zero occurred.")

def check_wifi_connection():
    try:
        # Try to connect to a public DNS server
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False
    
def create_urls (urls):
    def create_urls_and_append_to_catalog(base_url, catalog):
        # Extract the label from the base URL (assumed to be between "filter/" and the next "/")
        segments = base_url.split("/")
        label = segments[-2]  # Second-to-last segment (subcategory)

        # Create the additional URLs by appending different query parameters
        additional_queries = [
            "?s=N4IgrCBcoA5QTAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QzAGhDOkCMAGTBfHQ",
            "?s=N4IgrCBcoA5QLAGhDOkCMAGTBfHQ"
        ]
        
        # Create the list of full URLs (including the base one)
        new_urls = [base_url] + [base_url + query for query in additional_queries]

        # Ensure the label is present in the catalog and append URLs as a list
        if label in catalog:
            catalog[label].extend(new_urls)  # Append to existing list
        else:
            catalog[label] = new_urls  # Create new entry if label doesn't exist
    catalog = { }

    # Iterate over the list and call the function for each URL
    for url in urls:
        create_urls_and_append_to_catalog(url, catalog)

    return catalog