import text_preprocessing as txtp

# Preprocess text using default preprocess functions in the pipeline
text_to_process = 'Helllo, I am John Doe!!! My email is john.doe@email.com. Visit our website www.johndoe.com'
preprocessed_text = txtp.preprocess_text(text_to_process)
print(preprocessed_text)
# output: hello email visit website