"""
Using the github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
import text_preprocessing as txtp

# Preprocess text using default preprocess functions in the pipeline 
text_to_process = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
preprocessed_text = txtp.preprocess_text(text_to_process)
print(preprocessed_text)
# output: hello email visit website

# Preprocess text using custom preprocess functions in the pipeline 
preprocess_functions = [to_lower, keep_alpha_numeric, remove_email,remove_phone_number,remove_itemized_bullet_and_numbering,remove_stopword, remove_url, remove_punctuation, lemmatize_word]
preprocessed_text = txtp.preprocess_text(text_to_process, preprocess_functions)
print(preprocessed_text)
# output: helllo i am john doe my email is visit our website


## Testing with a file
file_path = "02-data/txts/accessories/2xx_M/text.txt"
file_path_out = "02-data/0-testing/02-nlp/text.txt"

# Ensure the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found at path: {file_path}")

# Read the text from the file
with open(file_path, 'r', encoding='utf-8') as file:
    text_to_process = file.read()

preprocess_functions = [to_lower, keep_alpha_numeric, remove_email,remove_phone_number,remove_itemized_bullet_and_numbering,remove_stopword, remove_url, remove_punctuation, lemmatize_word]
preprocessed_text = txtp.preprocess_text(text_to_process, preprocess_functions)

# Write the preprocessed text to a new file
with open(file_path_out, 'w', encoding='utf-8') as file:
    file.write(preprocessed_text)