import pdfplumber
import pandas as pd
from pdfplumber.utils import extract_text, get_bbox_overlap, obj_to_bbox
import tabulate



def process_pdf(pdf_path):
    pdf = pdfplumber.open(pdf_path)
    all_text = []
    tables_read = 0
    tables_missed = 0
    for page in pdf.pages:
        filtered_page = page
        chars = filtered_page.chars

        for table in page.find_tables():
            try:
                first_table_char = page.crop(table.bbox).chars[0] #Finding some tables without text inside
                #print("Found a table to analyze")
                tables_read = tables_read +1
            except IndexError:
                #print("There was a table without text")
                tables_missed = tables_missed +1

            filtered_page = filtered_page.filter(lambda obj: 
                get_bbox_overlap(obj_to_bbox(obj), table.bbox) is None
            )
            chars = filtered_page.chars

            df = pd.DataFrame(table.extract())
            df.columns = df.iloc[0]
            markdown = df.drop(0).to_markdown(index=False)

            chars.append(first_table_char | {"text": markdown})

        page_text = extract_text(chars, layout=True)
        all_text.append(page_text)

    pdf.close()
    print(f"Read {(tables_read/tables_missed)*100}% of tables")
    return "\n".join(all_text)

# Path to your PDF file
pdf_path = r"02-data/original_examples/pdf_file_4137.pdf"
extracted_text = process_pdf(pdf_path)

# Specify the filename
filename = "02-data/original_examples/pdf_file_4137_tables.txt"

# Save the variable content to a .txt file
with open(filename, "w") as file:
    file.write(extracted_text)