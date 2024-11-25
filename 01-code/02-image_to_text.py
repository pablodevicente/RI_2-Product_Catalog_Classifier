"""
a ver este scrip tiene que

coger cada una de las imagenes de los pdfs
    1. filtrar las que sean basura
        1.0 las imagenes que tengas menos de x pixeles
        1.1 las que tengan un ratio extraño (no 16x9) ni similar
        1.2 las imagenes que sean en un x% (95) de un solo color
        
para cada una de las imagenes, llamar a 1 (o 3) modelos y recoger la descripcion

"""

import os
from PIL import Image 
import logging
import shutil
from tqdm import tqdm

txt_path = "../02-data/txts/"
destination_path = "../02-data/0-testing/images"

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the minimum pixel threshold and acceptable aspect ratio range
MIN_PIXELS = 1000
MIN_ASPECT_RATIO = 0.2  # width / height. 200 px / 1000 px
MAX_ASPECT_RATIO = 5.0

# Function to check image properties and delete images based on pixel count and aspect ratio
def check_and_delete_images(image_path):
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
            
            # Check if the image meets the pixel count and aspect ratio requirements
            if pixel_count < MIN_PIXELS or not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                logger.debug(f"Deleting {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
                #os.remove(image_path)
                shutil.move(image_path, destination_path)
            else:
                logger.debug(f"Keeping {image_path} (Pixels: {pixel_count}, Aspect Ratio: {aspect_ratio:.2f})")
    except Exception as e:
        logger.debug(f"Error processing {image_path}: {e}")


# Get a list of labels (folders) in the root directory
labels = os.listdir(txt_path)

# Wrap the label loop with tqdm to show progress
for label in tqdm(labels, desc="Folders processed"):
    input_path = os.path.join(txt_path, label)

    # Traverse all subdirectories and files
    for root, _, files in os.walk(input_path):
        for file in files:
            # Check if the file is an image
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                image_path = os.path.join(root, file)
                check_and_delete_images(image_path)