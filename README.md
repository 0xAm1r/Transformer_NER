# Transformer NER

Named Entity Recognition (NER) using Transformer models, specifically DistilBERT, implemented with TensorFlow.

## Overview

This project implements a Named Entity Recognition system using DistilBERT for token classification. The model is trained on a custom dataset and can identify and classify named entities in text.

## Project Structure

```
Transformer_NER/
├── model/              # Model configuration and weights
│   ├── config.json    # Model configuration
│   └── tf_model.h5    # Model weights (not tracked in git)
├── tokenizer/         # Tokenizer files
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
├── ner.json           # Training dataset
├── utils.py           # Utility functions for data processing
├── Transformer_NER.ipynb  # Main training notebook
└── requirements.txt   # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Transformer_NER
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Open and run the `Transformer_NER.ipynb` notebook to train the model. The notebook includes:

- Data preprocessing and cleaning
- Entity extraction and labeling
- Tokenization and alignment
- Model training with DistilBERT
- Evaluation metrics

### Using the Model

After training, you can use the model for inference:

```python
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification
import tensorflow as tf

# Load tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')
model = TFDistilBertForTokenClassification.from_pretrained('model/', num_labels=12)

# Example inference
text = "Your text here"
inputs = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=512)
output = model(inputs).logits
predictions = tf.argmax(output, axis=2)
```

## Dataset

The training data is in `ner.json` format (JSONL - one JSON object per line) with the following structure:
- `content`: The text content
- `annotation`: List of annotations with labels and character positions

## Model Files

Large model files (`.h5`, `.pt`, `.pth`, etc.) are excluded from git via `.gitignore`. To use the model:

1. Train it using the notebook, or
2. Download pre-trained weights and place them in the `model/` directory

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries:
- TensorFlow 2.x
- Transformers (Hugging Face)
- Pandas
- NumPy

