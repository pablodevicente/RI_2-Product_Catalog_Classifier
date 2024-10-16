"""
Using the github repository https://github.com/berknology/text-preprocessing to preprocess text in python
"""
from text_preprocessing import preprocess_text
from text_preprocessing import to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word

# Preprocess text using default preprocess functions in the pipeline 
text_to_process = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
preprocessed_text = preprocess_text(text_to_process)
print(preprocessed_text)
# output: hello email visit website

# Preprocess text using custom preprocess functions in the pipeline 
preprocess_functions = [to_lower, remove_email, remove_url, remove_punctuation, lemmatize_word]
preprocessed_text = preprocess_text(text_to_process, preprocess_functions)
print(preprocessed_text)
# output: helllo i am john doe my email is visit our website