import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import numpy as np
from transformers import BitsAndBytesConfig
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

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
                logger.debug(f"Deleting {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
                os.remove(image_path)
                return True  # Image was deleted
            else:
                logger.debug(f"Keeping {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
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
    logging.debug(f"Prediction value for {image_path}: {val}")

    # Interpret the prediction based on a threshold (e.g., 0.5)
    if val >= 0.5:
        logging.debug(f"Image {image_path} is NOT a product (prediction: {val}). Deleting...")
        os.remove(image_path)
        logging.debug(f"Deleted image: {image_path}")
    else:
        logging.debug(f"Image {image_path} is a product (prediction: {val}). Keeping...")

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

    image = Image.open(image_path)

    try:
        # Call the model and get the text description
        text_description = model.answer_question(
            image, prompt, tokenizer, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3
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

    for root, _, files in os.walk(folder_path):
        logging.debug(f"Processing folder: {root}")

        # Filter only image files
        image_files = [file for file in files if file.lower().endswith(('.jpeg', '.jpg', '.png'))]

        # Only proceed if there are images in the folder
        if not image_files:
            logging.debug(f"No images found in {root}. Skipping file creation.")
            continue  # Skip this folder

        # Define the path for the output text file
        file_path = os.path.join(root, "images_to_txt.txt")
        logging.debug(f"Output file for this folder: {file_path}")

        # Open the file once
        with open(file_path, "w") as opened_file:
            logging.debug(f"Opened file for writing: {file_path}")

            # Pass the opened file as an argument
            kwargs["file_handle"] = opened_file

            for file in image_files:
                image_path = os.path.join(root, file)  # Full path to the image file
                logging.debug(f"Processing image: {image_path}")

                # Call the function with the image path and opened file
                try:
                    function(image_path, **kwargs)
                    logging.debug(f"Successfully processed image: {image_path}")
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")

    logging.debug(f"Finished processing all images in folder: {folder_path}")


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0


def main(pdf_path="../02-data/01-pdfs/accessories",
         pre_filter=True,
         classifier_filter=True,
         image_to_llm=True,
         classifier_model_path="../02-data/02-classifier/00-model/best_image_classifier.keras",
         llama_model="qresearch/llama-3.1-8B-vision-378",
         prompt_used="USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:",
         max_new_tokens=200):

    # Pre-filter images if requested
    if pre_filter:
        process_images(pdf_path, pre_filter)
        logging.info("Finished pre-filtering images")

    # Classifier filtering if requested
    if classifier_filter:
        classifier_model = load_model(classifier_model_path)
        process_images(pdf_path, classifier_filter, model=classifier_model)
        logging.info("Finished classifier filtering")

    # Generate descriptions with LLM if requested
    if image_to_llm:
        llama_instance, tokenizer_instance = import_model(llama_model)
        process_images(pdf_path, image_to_llm,
                       model=llama_instance,
                       tokenizer=tokenizer_instance,
                       prompt=prompt_used,
                       max_new_tokens=max_new_tokens)
        logging.info("Finished image description generation with LLM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from a PDF and apply various filters")
    parser.add_argument('--pdf_path', type=str, default="../02-data/01-pdfs/accessories",
                        help="Path to the PDF directory")
    parser.add_argument('--pre_filter', action='store_true', help="Enable pre-filtering of images")
    parser.add_argument('--classifier_filter', action='store_true', help="Enable classifier filtering of images")
    parser.add_argument('--image_to_llm', action='store_true', help="Enable description generation with LLM")
    parser.add_argument('--classifier_model_path', type=str,
                        default="../02-data/02-classifier/00-model/best_image_classifier.keras",
                        help="Path to the classifier model file")
    parser.add_argument('--llama_model', type=str, default="qresearch/llama-3.1-8B-vision-378",
                        help="Name of the llama model")
    parser.add_argument('--prompt_used', type=str,
                        default="USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:",
                        help="Prompt to use for the LLM")
    parser.add_argument('--max_new_tokens', type=int, default=200,
                        help="Maximum number of new tokens for LLM generation")

    args = parser.parse_args()

    main(pdf_path=args.pdf_path,
         pre_filter=args.pre_filter,
         classifier_filter=args.classifier_filter,
         image_to_llm=args.image_to_llm,
         classifier_model_path=args.classifier_model_path,
         llama_model=args.llama_model,
         prompt_used=args.prompt_used,
         max_new_tokens=args.max_new_tokens)