# Activate the virtual environment
# source .env/bin/activate

# Deactivate the virtual environment
# source .env/bin/deactivate

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the model and tokenizer
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

# Example sequences
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# Tokenize the sequences
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# Get model output (logits for classification)
output = model(**tokens)

# The output is logits, so we need to convert them to probabilities or labels
logits = output.logits
predicted_class_ids = torch.argmax(logits, dim=-1)

# Convert the predicted class IDs into labels (e.g., "positive" or "negative" for SST-2)
labels = ['negative', 'positive']  # For SST-2, it's a binary classification
predicted_labels = [labels[class_id] for class_id in predicted_class_ids]

# Print the sequences along with the predicted labels
for sequence, label in zip(sequences, predicted_labels):
    print(f"Sequence: {sequence} => Predicted label: {label}")