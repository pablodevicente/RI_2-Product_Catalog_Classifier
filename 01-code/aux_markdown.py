from spire.pdf.common import *
from spire.pdf import *
import pandas as pd
import pdfplumber

def extract_tables_from_pdf(pdf):
    """
    Extracts tables from each page in the PDF and converts them to markdown format.
    """
    all_tables = []
    for page in pdf.pages:
        for table in page.find_tables():
            return table

pdf_path = "/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/pdfs/accessories/2xx_M.pdf"
pdf = pdfplumber.open(pdf_path)
table = extract_tables_from_pdf(pdf)

doc = PdfDocument()
doc.LoadFromFile(pdf_path)


page_coord = table.page.bbox
table_coord = table.bbox
page = table.page

bookmarkTitle = "Bookmark-{0}".format(0 + 1)
bookmarkDest = PdfDestination(page, PointF(1.1, 1.1))
bookmark = doc.Bookmarks.Add(bookmarkTitle)


outputFile = "/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/0-testing/Bookmark_test.pdf"
doc.SaveToFile(outputFile)

# Close the document
doc.Close()