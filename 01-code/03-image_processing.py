
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Load the image
image_path = "/media/pablo/windows_files/00 - Master/05 - Research&Thesis/R2-Research_Internship_2/02-data/0-testing/fitz_im_extractor_examples/3m-cable-accessory-products-2013-electrical-products-catalog_page7_img2.jpeg"  # Replace with your image path

image = Image.open(image_path).convert("RGB")

# Preprocess the image and generate a description
inputs = processor(image, return_tensors="pt")
description_ids = model.generate(**inputs, max_new_tokens=50)
description = processor.decode(description_ids[0], skip_special_tokens=True)

print("Image description:", description)
