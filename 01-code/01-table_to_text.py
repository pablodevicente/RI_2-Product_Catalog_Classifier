import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the Llama model and tokenizer
model_name = "meta-llama/Llama-3.2-3B-Instruct"  # Update if different
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def read_table(filepath):
    # Read the table file
    return pd.read_csv(filepath, delimiter="|", skipinitialspace=True, engine="python")


def generate_sentence(row_data):
    # Format the prompt using all attributes and values in the row
    row_text = "\n".join(f"{attribute}: {value}" for attribute, value in row_data.items())
    prompt = (
        f"Generate a detailed description based on the following data:\n{row_text}\n\n"
        f"Description: "
    )

    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)

    # Decode and extract the generated description
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.split("Description:")[-1].strip()


def process_table(filepath):
    # Read the table data
    table_data = read_table(filepath)

    # Generate a sentence for each row in the table
    descriptions = []
    for index, row in table_data.iterrows():
        description = generate_sentence(row)
        descriptions.append(description)

    return descriptions


# Path to the input .txt file
file_path = "../02-data/0-testing/tables.txt"  # Replace with your .txt file path

# Process and generate descriptions
sentences = process_table(file_path)
for sentence in sentences:
    print(sentence)
