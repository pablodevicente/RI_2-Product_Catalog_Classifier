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
import sys

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0

# Create a StreamHandler with flush enabled
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.flush = sys.stdout.flush  # Explicit flush
logger.addHandler(stream_handler)

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
        return True
    else:
        logging.info(f"Image {image_path} is a product (prediction: {val}). Keeping...")
        return False

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

def process_images(folder_path, classifier_model, llama_instance, tokenizer_instance, prompt, max_new_tokens):
    """
    Processes image files in labeled folders, applying pre-filter, classifier, and LLM description generation.

    Parameters:
    - folder_path (str): Path to the root directory containing label folders.
    - classifier_model: Model for the classifier filter.
    - llama_instance: LLM model instance.
    - tokenizer_instance: Tokenizer for the LLM.
    - prompt (str): Prompt for LLM image description.
    - max_new_tokens (int): Max tokens for LLM output.
    """
    for root, _, files in os.walk(folder_path):
      logging.info(f"Processing folder: {root}")
      
      # Filter only image files
      image_files = [file for file in files if file.lower().endswith(('.jpeg', '.jpg', '.png'))]

      # Only proceed if there are images in the folder
      if not image_files:
          logging.info(f"No images found in {root}. Skipping file creation.")
          continue  # Skip this folder

      # Define the path for the output text file
      output_path = os.path.join(root, "images_to_txt.txt")
      logging.info(f"Output file for this folder: {output_path}")

      # Open the file once
      with open(output_path, "w") as opened_file:

          for file in image_files:
              image_path = os.path.join(root, file)
              logging.info(f"Processing image: {image_path}")

              # Apply pre-filter
              if pre_filter(image_path):
                  logging.info(f"Image {image_path} failed pre-filter. Skipping.")
                  continue

              # Apply classifier filter
              # if classifier_filter(image_path, model=classifier_model):
              #    logging.info(f"Image {image_path} failed classifier filter. Skipping.")
              #    continue

              # Generate description with LLM
              try:
                  image_to_llm(image_path, file_handle=opened_file, model=llama_instance, tokenizer=tokenizer_instance, prompt=prompt, max_new_tokens=max_new_tokens)
                  logging.info(f"Successfully processed image with LLM: {image_path}")
              except Exception as e:
                  logging.error(f"Error generating LLM description for image {image_path}: {e}")

    logging.info(f"Finished processing all images in folder: {folder_path}")

def main(pdf_path, classifier_model_path, llama_model, prompt_used, max_new_tokens):
    # Load models once
    classifier_model = load_model(classifier_model_path)
    llama_instance, tokenizer_instance = import_model(llama_model)

    process_images(pdf_path, classifier_model, llama_instance, tokenizer_instance, prompt_used, max_new_tokens)
    logging.info("Finished processing all images with pre-filter, classifier, and LLM")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images from a PDF and apply various filters")
    parser.add_argument('--pdf_path', type=str,
                        default="../02-data/00-testing/03-demo/",
                        help="Path to the PDF directory")
    parser.add_argument('--classifier_model_path', type=str,
                        default="../02-data/02-classifier/00-model/best_image_classifier.keras",
                        help="Path to the classifier model file")
    parser.add_argument('--llama_model', type=str,
                        default="qresearch/llama-3.1-8B-vision-378",
                        help="Name of the llama model")
    parser.add_argument('--prompt_used', type=str,
                        default="USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:",
                        help="Prompt to use for the LLM")
    parser.add_argument('--max_new_tokens', type=int,
                        default=200,
                        help="Maximum number of new tokens for LLM generation")

    args = parser.parse_args()

    main(pdf_path=args.pdf_path,
         classifier_model_path=args.classifier_model_path,
         llama_model=args.llama_model,
         prompt_used=args.prompt_used,
         max_new_tokens=args.max_new_tokens)