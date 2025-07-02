# config.py

# ─────────────────────────────
# PDF Download Configuration
# ─────────────────────────────
BASE_PATH = "../../02-data/01-pdfs/"
URL_PATH = "../../02-data/01-pdfs/digikeys_urls.txt"

# ─────────────────────────────
# Models used for table-to-text
# ─────────────────────────────
TABLE_GPT = "tablegpt/TableGPT2-7B"
QWN = "Qwen/Qwen2.5-7B-Instruct"
# Prompt template is hardcoded, look into aux_extract_pdf/generate_tablegpt_description and aux_extract_pdf/generate_qwn_description

# ─────────────────────────────
# Variables used for Image-to-text
# ─────────────────────────────
IMAGE_CLASSIFIER = "../../02-data/02-classifier/00-model/best_image_classifier.keras"
LLAMA_VISION = "qresearch/llama-3.1-8B-vision-378"
PROMPT = "USER: <image>\nDescribe in a technical manner the elements in the image\nASSISTANT:"
MAX_NEW_TOKENS = 200

# ─────────────────────────────
# Variables used for TTI.txt
# ─────────────────────────────
CLEANUP = False

# ─────────────────────────────
# Variables used for training image classifier
# ─────────────────────────────
LOG_CLASSIFIER = '../../02-data/02-classifier/00-model/tensorflow_rendezvous_logs.txt'
TRAIN_DIR = '../../02-data/02-classifier/train'
VALID_DIR = '../../02-data/02-classifier/valid'
TEST_DIR = '../../02-data/02-classifier/test'
MODEL_CHECKPOINT = '../02-data/02-classifier/00-model/best_image_classifier.keras' ## same as image_classifier route, but i will leave them separate
CLASSIFIER_SAVE = '../02-data/02-classifier/00-model/simple_image_classifier.keras' ## could be streamlined, but rather save them differently