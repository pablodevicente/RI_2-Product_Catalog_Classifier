from PIL import Image
import pytesseract
import sys
import os

def extract_text_from_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return None

    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Failed to process image: {e}")
        return None

if __name__ == "__main__":


    path = '../blas.png'
    text = extract_text_from_image(path)

    if text:
        print("\nExtracted Text:\n")
        print(text)
    else:
        print("No text found or failed to extract.")