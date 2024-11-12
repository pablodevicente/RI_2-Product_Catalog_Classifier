# PDF Download and Extraction Pipeline

This project consists of Python scripts for downloading PDFs from DigiKey's product pages, organizing them, and extracting text, images, and tables from each PDF. Below is a description of each script and its usage.

---

## Scripts Overview

### 00-download_pdfs.py

**Purpose**:  
Downloads PDFs from DigiKey product pages and cleans any corrupted files (e.g., files that aren't actual PDFs).

**Process**:
1. Reads URLs from `urls.txt`, which contains links to various product categories on DigiKey (e.g., [Controller Accessories](https://www.digikey.com/en/products/filter/controllers/controller-accessories/816)).
2. Generates links for each page in the product category, with pagination in the format:  
   `link1,page1`, `link1,page2`, `link1,page3`, etc.  
   Each page is assumed to contain up to 100 PDFs.
3. Downloads the PDFs to the specified output path, handling errors as necessary.

**Dependencies**:  
Uses helper functions in `aux_download_pdfs.py`.

---

### 01-extract_pdf.py

**Purpose**:  
Processes downloaded PDFs, extracting their text, images, and tables into a structured folder format.

**Process**:
1. Reads from the PDF directory.
2. Creates a corresponding directory structure for text output:
   ```
          data
          ├── pdfs
          │   ├── label1..n
          │        ├── pdf1..n
          └── txt
              ├── label1..n
                   ├── pdf1..n
                        ├── text
                        ├── images1..n
                        ├── tables
   ```

4. Each PDF file’s content is saved separately, with text, images, and tables extracted into their respective subfolders.

**Dependencies**:  
Uses helper functions in `aux_extract_pdf.py`.

---


01-image to text

01 table to text
