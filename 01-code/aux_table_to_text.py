import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re


def process_table(filepath):
    table_data = pd.read_csv(filepath, delimiter="|", skipinitialspace=True, engine="python")

    # Format each row as "column_name: column_value, column_name: column_value, ..."
    formatted_rows = []
    for _, row in table_data.iterrows():
        row_data = ", ".join(f"{col_name.strip()}: {str(value).strip()}" for col_name, value in row.items())
        formatted_rows.append(row_data)

    return formatted_rows


# Function to filter out unwanted columns
def filter_columns(row):
    # Split by commas to get each column separately
    columns = row.split(", ")
    # Keep only columns that don't start with "Unnamed" and aren't set to "nan" or similar placeholder values
    filtered_columns = [
        col for col in columns
        if not col.startswith(
            "Unnamed") and "nan" not in col.lower() and "-----------" not in col and "---------" not in col
    ]
    # Join back the filtered columns
    return ", ".join(filtered_columns)


def read_table_from_file(filename):
    # Read the table from a file, splitting each line into header and value based on the first colon
    with open(filename, 'r') as file:
        lines = file.readlines()

    data = {}
    for line in lines:
        # Skip separator lines if present
        if line.startswith('|:') or line.startswith('|---'):
            continue

        # Extract header and value by splitting at the first colon found
        match = re.match(r'\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|', line)
        if match:
            header = match.group(1).strip()
            value = match.group(2).strip()
            data[header] = value

    # Convert to DataFrame for transformation
    table = pd.DataFrame([data])
    return table


def transform_table_to_dict(table):
    # Initialize the list that will hold the final transformed data
    transformed_data = []

    # Create header row
    header_row = []
    for col_name in table.columns:
        header_row.append({
            'column_span': 1,
            'is_header': True,
            'row_span': 1,
            'value': col_name
        })
    transformed_data.append(header_row)

    # Create data rows
    for index, row in table.iterrows():
        data_row = []
        for col_name in table.columns:
            data_row.append({
                'column_span': 1,
                'is_header': False,
                'row_span': 1,
                'value': row[col_name]
            })
        transformed_data.append(data_row)

    return transformed_data


def serialize_table(table_data):
    # Initialize the serialized output with the table tags
    serialized_output = "<s> <table>"

    # Process the header row
    headers = table_data[0]  # assuming the first row is always the header
    serialized_output += " <row>"
    for cell in headers:
        serialized_output += f" <c> {cell['value']} </c>"
    serialized_output += " </row>"

    # Process data rows
    for row in table_data[1:]:
        serialized_output += " <row>"
        for cell, header_cell in zip(row, headers):
            col_header = header_cell['value']
            serialized_output += f" <c> {cell['value']} <col_header> {col_header} </col_header> </c>"
        serialized_output += " </row>"

    # Close table and document tags
    serialized_output += " </table> </s>"

    return serialized_output

