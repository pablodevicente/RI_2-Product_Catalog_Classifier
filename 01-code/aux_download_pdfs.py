import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm  # Correct import
import fitz  # PyMuPDF
import pandas as pd

def download_pdfs_from_page(label, url, save_folder, df_statistics):
    updated_df = df_statistics
    
    #if the path allready exists, we have downloaded everything in that label.
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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

        # statistics
        total_processed = len(pdf_links)
        total_success = 0
        total_failed = 0
        total_exceptions = 0

        # Set up the progress bar
        with tqdm(total=total_processed, desc="Downloading PDFs", unit="pdf") as pbar:
            for pdf_link in pdf_links:
                try:
                    pdf_response = requests.get(pdf_link)
                    if pdf_response.status_code == 200:
                        pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                        with open(pdf_filename, 'wb') as f:
                            f.write(pdf_response.content)
                        total_success += 1  # Increment successful downloads
                    else:
                        total_failed += 1  # Increment failed downloads

                except Exception as e:
                    total_exceptions += 1  # Increment exceptions count
                
                pbar.update(1)  # Update progress bar regardless
        
        # Print download statistics
        print(f"Total: {total_processed} / 100 | "
            f"Downloaded: {total_success} ({(total_success/total_processed)*100:.2f}%) | "
            f"Failed: {total_failed} ({(total_failed/total_processed)*100:.2f}%) | "
            f"Exceptions: {total_exceptions} ({(total_exceptions/total_processed)*100:.2f}%)")

        # Update the DataFrame with the statistics
        if label in df_statistics['Label'].values:
            # Update existing row
            df_statistics.loc[df_statistics['Label'] == label, 'Total_found'] += total_processed
            df_statistics.loc[df_statistics['Label'] == label, 'Downloaded_Valid'] += total_success
            df_statistics.loc[df_statistics['Label'] == label, 'Downloaded_Invalid'] += total_failed
            df_statistics.loc[df_statistics['Label'] == label, 'Downloaded_Exceptions'] += total_exceptions
        else:
            # Create a new row
            new_row = {
                'Label': label,
                'Total_found': total_processed,
                'Downloaded_Valid': total_success,
                'Downloaded_Invalid': total_failed,
                'Downloaded_Exceptions': total_exceptions,
                'Cleanup_Valid': 0,  
                'Cleanup_Invalid': 0,
                'Cleanup_Exceptions': 0
            }
            # Convert the new_row to a DataFrame for concatenation
            new_row_df = pd.DataFrame([new_row])

            # Append the new row using pd.concat() and reassign df_statistics
            updated_df = pd.concat([df_statistics, new_row_df], ignore_index=True)
    return updated_df  
        

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


def check_pdfs_in_folder(label, folder_path, df_statistics):
    total_files = 0
    valid_files = 0
    invalid_files = 0
    total_exceptions = 0  # Initialize exceptions count (if needed in the future)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):  # Check if the file is a PDF
            total_files += 1
            file_path = os.path.join(folder_path, filename)
            if is_pdf_valid(file_path):
                valid_files += 1
            else:
                invalid_files += 1

    # Calculate statistics
    total_processed = total_files
    total_success = valid_files
    total_failed = invalid_files

    # Print the statistics in the desired format
    print(f"\nTotal: {total_processed} | "
          f"Valid: {total_success} ({(total_success/total_processed)*100:.2f}%) | "
          f"Invalid: {total_failed} ({(total_failed/total_processed)*100:.2f}%) | "
          f"Exceptions: {total_exceptions} ({(total_exceptions/total_processed)*100:.2f}%)")

    # Update the DataFrame with the statistics
    if label in df_statistics['Label'].values:
        # Update existing row
        df_statistics.loc[df_statistics['Label'] == label, 'Cleanup_Valid'] += valid_files
        df_statistics.loc[df_statistics['Label'] == label, 'Cleanup_Invalid'] += invalid_files
        df_statistics.loc[df_statistics['Label'] == label, 'Cleanup_Exceptions'] += total_exceptions
    else:
        # Create a new row
        new_row = {
            'Label': label,
            'Total_found': 0,  # Initialize if needed or use your desired value
            'Downloaded_Valid': 0,
            'Downloaded_Invalid': 0,
            'Downloaded_Exceptions': 0,
            'Cleanup_Valid': valid_files,
            'Cleanup_Invalid': invalid_files,
            'Cleanup_Exceptions': total_exceptions
        }
        df_statistics = df_statistics.append(new_row, ignore_index=True)

    return df_statistics  # Return the updated DataFrame