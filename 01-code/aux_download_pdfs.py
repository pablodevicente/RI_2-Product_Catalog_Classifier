import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm  # Correct import
import fitz  # PyMuPDF
import socket

def download_pdfs_from_page(urls, save_folder):
    
    #if the path allready exists, we have downloaded everything in that label.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

        for url in urls:

            # Send request to fetch content of the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Failed to retrieve webpage: {response.status_code}")
                return

            # Parse the webpage content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all <a> tags with href attributes that contain '.pdf'
            pdf_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '.pdf' in href:
                    # Handle relative URLs that start with //
                    if href.startswith('//'):
                        href = 'https:' + href
                    pdf_url = urljoin(url, href)
                    pdf_links.append(pdf_url)

            # Statistics
            total_processed = len(pdf_links)
            total_success = 0
            total_failed = 0
            total_exceptions = 0

            # Set up the progress bar
            with tqdm(total=total_processed, desc="Downloading PDFs from url", unit="pdf") as pbar:
                for pdf_link in pdf_links:
                    try:
                        pdf_response = requests.get(pdf_link)
                        if pdf_response.status_code == 200:
                            pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                            with open(pdf_filename, 'wb') as f:
                                f.write(pdf_response.content)
                            total_success += 1 
                        else:
                            total_failed += 1  
                    except Exception as e:
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
        
        # Create the list of full URLs
        new_urls = [base_url] + [base_url + query for query in additional_queries]

        # Check if the label exists in the catalog
        if label in catalog:
            # Append the new URLs to the existing list
            catalog[label].extend(new_urls)
        else:
            # Create a new entry for the label if it doesn't exist
            catalog[label] = new_urls


    catalog = { }

    # Iterate over the list and call the function for each URL
    for url in urls:
        create_urls_and_append_to_catalog(url, catalog)

    return catalog