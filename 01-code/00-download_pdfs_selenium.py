from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import os
import requests
from urllib.parse import urljoin
import time
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# Path to save data
data_path = "02-data/pdfs/"

# Path to the chromedriver executable
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.binary_location = "/usr/bin/google-chrome"  # Change this to the correct path
chrome_options.setBinary("/usr/bin/google-chrome")

chrome_path = "/media/pablo/windows_files/00 - Master/05 - Research&Thesis/03-chromedriver/chromedriver-linux64/chromedriver"
service = Service(chrome_path, service_log_path="/03-chromedriver/chromedriver.log")

driver = webdriver.Chrome(service=service, options=chrome_options)


def setup_save_folder(save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    print(f"Folder '{save_folder}' is ready.")


def download_pdfs_from_page(soup, save_folder, headers):
    pdf_links = []
    
    # Find all <a> tags with href attributes that contain '.pdf'
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '.pdf' in href:
            if href.startswith('//'):
                href = 'https:' + href
            pdf_url = urljoin(driver.current_url, href)
            pdf_links.append(pdf_url)

    # Download each PDF
    for pdf_link in pdf_links:
        try:
            pdf_response = requests.get(pdf_link, headers=headers)
            if pdf_response.status_code == 200:
                pdf_filename = os.path.join(save_folder, os.path.basename(pdf_link))
                with open(pdf_filename, 'wb') as f:
                    f.write(pdf_response.content)
                print(f"Downloaded: {pdf_filename}")
            else:
                print(f"Failed to download: {pdf_link}")
        except Exception as e:
            print(f"Error downloading {pdf_link}: {e}")


def go_to_next_page(driver):
    try:
        next_button = driver.find_element(By.CSS_SELECTOR, '[aria-label="Next Page"]')
        next_button.click()
        time.sleep(2)  # Wait for the next page to load
        return True
    except Exception as e:
        print("No more pages or unable to click 'Next':", e)
        return False


def download_pdfs_from_digikey(total_pages, save_folder=data_path):
    
    setup_save_folder(save_folder)

    # Load the first page
    driver.get('https://www.digikey.com/en/products/filter/controllers/controller-accessories/816?s=N4IgrCBcoA5QjAGhDOkBMYC%2BWg')

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    for page in range(1, total_pages + 1):
        
        # Get the HTML content
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Download PDFs from the current page
        download_pdfs_from_page(soup, save_folder, headers)

        # Navigate to the next page
        if not go_to_next_page(driver):
            break

    # Close the browser after you're done
    driver.quit()