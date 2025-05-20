# Semantic Role Labelling with BERT

This project implements a **Semantic Role Labelling (SRL)** system using a fine-tuned BERT model in PyTorch. The task follows PropBank-style annotations and is framed as a token-level sequence labeling problem using B-I-O (Begin, Inside, Outside) tags.

## 📚 Overview

For each sentence and its predicate(s), the model predicts argument spans as B-I-O labels corresponding to semantic roles. The model uses **BERT** (via Huggingface's Transformers library) to compute contextualized token embeddings. A custom classification head is added on top of BERT to output role labels for each token.

We exploit BERT's **segment embeddings** to indicate the predicate's position within a sentence. Tokens belonging to the predicate receive segment ID 1, while all others receive segment ID 0.

## ✨ Example

| Sentence                                     | Predicate     | BIO Tags                          |
|----------------------------------------------|---------------|-----------------------------------|
| The judge scheduled to preside over his trial | schedule.01   | B-ARG1 I-ARG1 B-V B-ARG2 ...      |
| The judge ... was removed from the case      | remove.01     | B-ARG1 I-ARG1 ... B-V B-ARG2 ...  |

## 📁 Files Included

- `srl_with_bert.ipynb`: Jupyter notebook with code, comments, and outputs.
- `srl_with_bert.py`: Script version of the notebook.
- `requirements.txt`: Python dependencies for the project.

## 📦 Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
