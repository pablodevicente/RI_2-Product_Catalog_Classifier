import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from transformers import BitsAndBytesConfig
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

txt_path = "../02-data/txts/"
destination_path = "../02-data/0-testing/images"

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0

def pre_filter(image_path):
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

def classifier_filter(image_path,model):
    ## CALL endpoint for classifier --> look into aux_classifier_training

    img1 = image.load_img(image_path, target_size=(150, 150))

    # Convert image to array and add batch dimension
    Y = image.img_to_array(img1)
    X = np.expand_dims(Y, axis=0)

    # Make prediction
    val = model.predict(X)[0][0]  # Get the scalar value from the output
    print(val)

    # Interpret the prediction based on a threshold (e.g., 0.5) ## >>0.5 == not_product
    if val >= 0.5:
        logger.debug(f"img : {image_path} not a product, deleting")
        os.remove(image_path)
    else: ## product
        logger.debug(f"img : {image_path} is a product")
        pass

def image_to_llm(image_path, file_handle, model, tokenizer, prompt, max_new_tokens):
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
    - txt_path (str): Path to the root directory containing label folders.
    - function (callable): Function to apply to each image.
    - kwargs: Additional arguments for the processing function.
    """
    folders = os.listdir(folder_path)
    for subfolder in folders:
        subfolder_path = os.path.join(txt_path, subfolder)
        for root, _, files in os.walk(subfolder_path):
            output_file = None
            try:
                for file in files:
                    if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                        image_path = os.path.join(root, file)

                        # Open output file if processing with image_to_llm
                        if output_file is None and "file_handle" in kwargs:
                            output_file_path = os.path.join(root, "images_to_txt")
                            output_file = open(output_file_path, "w")

                        # Pass arguments dynamically to the function
                        if "file_handle" in kwargs:
                            kwargs["file_handle"] = output_file
                        function(image_path, **kwargs)
            finally:
                if output_file:
                    output_file.close()

# Pre-filter images
process_images(txt_path, pre_filter)

# Classifier filtering
# Load the model from the file --> look into aux_train_classifier.ipynb
classifier_model = load_model("../02-data/02-classifier/model.keras")
process_images(txt_path, classifier_filter,model = classifier_model)

# Import llama model
llama_model = "qresearch/llama-3.1-8B-vision-378"
llama_instance, tokenizer_instance = import_model(llama_model)
prompt_used = "USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:"

# Generate descriptions with the LLM
process_images(txt_path, image_to_llm,
    file_handle=None,
    model=llama_instance,
    tokenizer=tokenizer_instance,
    prompt=prompt_used,
    max_new_tokens=200
)