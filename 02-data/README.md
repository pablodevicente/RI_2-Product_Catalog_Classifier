## Data

This section describes the example data and directory layout. Large or proprietary files (e.g. full database dumps, pretrained embeddings) are **not** included—see instructions below for how to generate or download them yourself.

### Directory Structure

- **02-data/**
  - **00-testing/**  
    - `00-testing.tar.gz/` – Processed example classes”
  - **01-pdfs/**  
    - `accessories/` – Raw & processed PDF examples for “accessories”  
    - `Digikeys download` – URLs & metadata for DigiKey PDF downloads
  - **02-image_classifier.tar.gz**  
    - `train/` – Images & labels for training the image classifier  
    - `test/` – Images & labels for testing  
    - `validation/` – Images & labels for validation  
  - **03-vsm/** – _Not included_  
    - `01-Word2Vec/` – Pretrained & finetuned Word2Vec binaries  
    - `02-Glove/` – GloVe embeddings  
    - `03-Fasttext/` – FastText embeddings  
  - **04-classifier/** – _Not fully included_  
    - `classifiers-4-150.csv` – Classifier data  
---

### What’s Included

- **`00-testing/`**  
  Minimal, processed examples—ideal for quick experimentation or sanity checks. These classes are intentionally small and incomplete.
  
- **`01-pdfs/`**  
  - **`accessories/`**: A fully processed PDF example for the “accessories” class.  
  - **`Digikeys download`**: A manifest of DigiKey URLs; see the download script under `01-code/0_Download_pdfs/`.

- **`02-image_classifier.tar.gz`**  
  Contains the train/test/validation splits for the image‑based classifier.

---

### What’s **Not** Included

- Full database of PDFs or extracted text versions  
- Generated VSM embeddings under `03-vsm/`  
- Pretrained / finetuned Word2Vec, FastText, and GloVe model binaries
- Full classifier results and versions

---

### How to Reproduce Missing Pieces
**Download PDFs**  
```bash
   python download_pdfs.py --manifest "../01-pdfs/Digikeys download"
 ```

**Generate Text & Embeddings**

**Train Classifiers**
```bash
  python train_classifiers.py --config classifier_config.yml
```

Once these steps complete, you’ll have the full dataset, embeddings, and classifier results ready for analysis.