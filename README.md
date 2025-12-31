# Transformer NER

Named Entity Recognition (NER) using Transformer models, specifically DistilBERT, implemented with TensorFlow.

## Overview

This project implements a Named Entity Recognition system using DistilBERT for token classification. The model is trained on a custom dataset and can identify and classify named entities in text.

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
