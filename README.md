# Semantic Role Labelling with BERT

This project implements a BERT-based sequence labeling system for PropBank-style semantic role labeling, fine-tuned on OntoNotes 5.0 SRL annotations.


## Overview

This project implements a Semantic Role Labeling (SRL) system using PyTorch and Huggingface’s BERT. The task is modeled as sequence labeling over BIO-formatted tags with special treatment of predicate positions via segment embeddings.

Project stages:
- Data Preparation – Format OntoNotes SRL data into BIO-tagged .tsv files.
- Modelling – Fine-tune BERT with predicate-aware segment embeddings.
- Training & Evaluation – Mask padding tokens; evaluate per-predicate.
- Utilities – Includes notebook, script, and requirements list.

## Project Structure
```text
.
├── data/  # Folder to hold unzipped OntoNotes SRL dataset files
│   ├── propbank_train.tsv  # BIO-tagged training data
│   ├── propbank_dev.tsv  # BIO-tagged dev data
│   ├── propbank_test.tsv  # BIO-tagged test data
│   └── role_list.txt  # List of semantic role labels
├── srl_bert.ipynb  # Jupyter notebook for training and evaluation
├── srl_bert.py  # Source code for core project scripts
├── ontonotes_srl.zip  # Compressed archive of formatted OntoNotes SRL data (not public)
├── requirements.txt  # Project dependencies
└── README.md
```

## Files

- `srl_bert.ipynb`: Interactive Jupyter notebook for training and evaluation.
- `srl_bert.py`: Script version for non-interactive use.
- `ontonotes_srl.zip`: Archive containing formatted SRL datasets.
- `propbank_train.tsv`, `propbank_dev.tsv`, `propbank_test.tsv`: BIO-tagged data.
- `role_list.txt`: List of semantic role labels used.
-  `requirements.txt`: Project dependencies.


## Dataset

This project uses the English portion of OntoNotes 5.0 with PropBank-style annotations.
Note: The dataset is not publicly included due to LDC licensing.

## Setup

1. Clone the repository.
2. Create a virtual environment (optional).
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Unzip `ontonotes_srl.zip`.
5. Check GPU availability with:
```bash
import torch
print(torch.cuda.is_available())
```

## Usage
### Jupyter Notebook
Run all cells in `srl_bert.ipynb` for end-to-end training and evaluation.
### Python Script
Use the script version:
```bash
python srl_bert.py
```

## Notes
- Assumes predicates are known (no predicate identification).
- Predicate positions are marked using segment embeddings.
- BIO-tag alignment handles subword tokenization correctly.
- Padding tokens are masked during loss calculation.
- Possible extensions: Add predicate identification, support span-based labelling, incorporate newer models (e.g., RoBERTa, DeBERTa), or adapt to multilingual SRL.

