import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
from transformers import BitsAndBytesConfig
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def pre_filter(image_path, **kwargs):
    """
    Checks image properties and deletes images that do not meet specified criteria.

    Parameters:
    - image_path (str): Path to the image file.

    Criteria:
    - Minimum Pixels: The image must have at least MIN_PIXELS.
    - Aspect Ratio: The width-to-height ratio must be between MIN_ASPECT_RATIO and MAX_ASPECT_RATIO.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            pixel_count = width * height
            aspect_ratio = width / height

            if pixel_count < MIN_PIXELS or not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                logger.info(f"Deleting {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
                os.remove(image_path)
                return True  # Image was deleted
            else:
                logger.info(f"Keeping {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
                return False  # Image was kept
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def classifier_filter(image_path, **kwargs):
    """
    Classifies an image using a provided model and filters it based on the prediction.

    Parameters:
    - image_path (str): Path to the image file.
    - kwargs: Additional arguments, including the 'model' for classification.
    """
    model = kwargs.get('model')
    if model is None:
        raise ValueError("The 'model' argument is required for classifier_filter.")

    # Load and preprocess the image
    img1 = image.load_img(image_path, target_size=(150, 150))
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)

    # Make prediction
    val = model.predict(X, verbose=0)[0][0]  # Get the scalar value from the output. verbose=0 restricts output
    logging.info(f"Prediction value for {image_path}: {val}")

    # Interpret the prediction based on a threshold (e.g., 0.5)
    if val >= 0.5:
        logging.info(f"Image {image_path} is NOT a product (prediction: {val}). Deleting...")
        os.remove(image_path)
        logging.info(f"Deleted image: {image_path}")
    else:
        logging.info(f"Image {image_path} is a product (prediction: {val}). Keeping...")

def image_to_llm(image_path, **kwargs):
    """
    Processes an image using a language model and writes the description to the file.

    Parameters:
    - image_path (str): Path to the image file.
    - file_handle (file object): Handle to the output file.
    - model: The LLM model instance.
    - tokenizer: The tokenizer for the LLM.
    - prompt (str): Prompt to pass to the model.
    - max_new_tokens (int): Maximum tokens for the response.
    """
    # Extract the arguments from kwargs
    model = kwargs.get('model')
    file_handle = kwargs.get('file_handle')
    tokenizer = kwargs.get('tokenizer')
    prompt = kwargs.get('prompt')
    max_new_tokens = kwargs.get('max_new_tokens', 200)  # Default to 200 if not provided

    try:
        # Call the model and get the text description
        text_description = model.answer_question(
            image_path, prompt, tokenizer, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3
        )
        file_handle.write(f"{text_description}\n\n")
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")

def import_model(model_path):
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["mm_projector", "vision_model"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
    )

    return model,tokenizer

def process_images(folder_path, function, **kwargs):
    """
    Processes image files in labeled folders, applying a given function to each image.

    Parameters:
    - folder_path (str): Path to the root directory containing label folders.
    - function (callable): Function to apply to each image.
    - kwargs: Additional arguments for the processing function.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")

    logging.info(f"Starting image processing in folder: {folder_path}")

    for root, _, files in os.walk(folder_path):
        logging.info(f"Processing folder: {root}")

        # Define the path for the output text file (only once per document)
        file_path = os.path.join(root, "images_to_txt.txt")
        logging.info(f"Output file for this folder: {file_path}")

        # Open the file once
        with open(file_path, "w") as opened_file:
            logging.info(f"Opened file for writing: {file_path}")

            # Pass the opened file as an argument
            kwargs["file_handle"] = opened_file

            for file in files:
                # Check if the file is an image (JPEG, JPG, or PNG)
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    image_path = os.path.join(root, file)  # Full path to the image file
                    logging.info(f"Processing image: {image_path}")

                    # Call the function with the image path and opened file
                    try:
                        function(image_path, **kwargs)
                        logging.info(f"Successfully processed image: {image_path}")
                    except Exception as e:
                        logging.error(f"Error processing image {image_path}: {e}")
                else:
                    logging.warning(f"Skipping non-image file: {file}")

    logging.info(f"Finished processing all images in folder: {folder_path}")


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0

pdf_path = "../02-data/01-pdfs/00-testing"

# Pre-filter images
process_images(pdf_path, pre_filter)

# Classifier filtering
classifier_model = load_model("../02-data/02-classifier/model.keras") # Load the model from the file --> look into aux_train_classifier.ipynb
process_images(pdf_path, classifier_filter, model=classifier_model)

# Import llama model
llama_model = "qresearch/llama-3.1-8B-vision-378"
llama_instance, tokenizer_instance = import_model(llama_model)
prompt_used = "USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:"

# Generate descriptions with the LLM
process_images(pdf_path, image_to_llm,
    model=llama_instance,
    tokenizer=tokenizer_instance,
    prompt=prompt_used,
    max_new_tokens=200
)