import os
from PIL import Image 
import logging
import shutil
from tqdm import tqdm
import requests
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers import BitsAndBytesConfig



txt_path = "../02-data/txts/"
destination_path = "../02-data/0-testing/images"

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0

# Function to check image properties and delete images based on pixel count and aspect ratio -- pre-filtering
def pre_filter(image_path,file_handle):
    """
    Opens an image file, checks its pixel count and aspect ratio, 
    and deletes it if it doesn't meet specified criteria.

    Parameters:
    - image_path (str): Path to the image file to be checked.

    Criteria:
    - Minimum Pixels: The image must have at least MIN_PIXELS.
    - Aspect Ratio: The width-to-height ratio must fall between MIN_ASPECT_RATIO and MAX_ASPECT_RATIO.

    If the image does not meet the criteria, it is deleted. 
    A debug log is created for each action taken.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            pixel_count = width * height
            aspect_ratio = width / height
            deleted = 0

            # Check if the image meets the pixel count and aspect ratio requirements
            if pixel_count < MIN_PIXELS or not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                logger.debug(f"Deleting {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
                os.remove(image_path)
                #shutil.move(image_path, destination_path)
                deleted = 1
            else:
                logger.debug(f"Keeping {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
    except Exception as e:
        logger.debug(f"Error processing {image_path}: {e}")

    return deleted

def classifier_filter(image_path, file_handle):
    ## CALL endpoint for classifier --> look into aux_classifier_training
    ## opt1 : train and call deployed model
    ## opt2 : call endpoint, limited

    ##delete images that are classified as not_product
    pass

def image_to_llm(image_path, file_handle):
    """
    Calls the model to process the image and writes the result into the file handle.
    """
    # Simulate calling the Llama model and getting the text description
    # Replace the line below with your actual model call
    text_description =  llama_model.answer_question(
        image_path, prompt , tokenizer, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.3
    )
    # Write the result to the provided file handle
    file_handle.write(f"{text_description}\n\n")

    ## CALL LlaVa -> get text description2
    ## CALL Florence -> get text description3

    # Call Llama -> concatenate all descriptions

    return file_handle

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


def process_images(txt_path, function):
    """
    Processes image files in labeled folders, applying a given function to each image.

    Args:
        txt_path (str): Path to the root directory containing label folders.
    """
    # Get a list of labels (folders) in the root directory
    folders = os.listdir(txt_path)

    # Wrap the label loop with tqdm to show progress
    for folder in folders:
        folder_path = os.path.join(txt_path, folder)

        # Traverse all subdirectories and files
        for root, _, files in os.walk(folder_path):
            output_file = None  # Initialize output file handle
            try:
                for file in files:
                    # Check if the file is an image
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        image_path = os.path.join(root, file)

                        # Create the output file if not already created for this subdirectory
                        if output_file is None:
                            output_file_path = os.path.join(root, f"image_to_txt")
                            output_file = open(output_file_path, "w")  # Open the file in write mode

                        # Call the processing function and write its result to the output file
                        processed_content = function(image_path, output_file)
            finally:
                # Ensure the output file is closed properly
                if output_file:
                    output_file.close()
'''
1. Filters out images : size(img) < threshold || ratio < x
2. Filters out images not classified as products
3. Calls LLM to describe remaining images
'''
process_images(txt_path,pre_filter)
process_images(txt_path,classifier_filter)

llama_model = "qresearch/llama-3.1-8B-vision-378"
model,tokenizer = import_model(llama_model)
max_new_tokens = 200
prompt = "USER: <image>\nDescribe in a tecnhical manner the elements in the image\nASSISTANT:"

process_images(txt_path,image_to_llm)