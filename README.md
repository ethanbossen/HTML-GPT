# Hugging Face Transformer Model Integration

This repository demonstrates how to interact with Hugging Face models using the transformers library for various NLP tasks, including text generation and model fine-tuning. The project consists of two main scripts that showcase different aspects of working with Hugging Face transformers.

## Requirements

To run the code, ensure that you have the following libraries installed:

```pip install transformers datasets torch```

## Scripts Overview

 1. Training_gpt2.py - Fine-tuning a Language Model

This script fine-tunes a pre-trained GPT-2 model on a custom dataset (in this case, band information from the metal-archives dataset). The script does the following:
	•	Loads the metal-archives-bands dataset.
	•	Preprocesses the dataset into a combined text format with band name, genre, and theme.
	•	Tokenizes the dataset.
	•	Fine-tunes a pre-trained GPT-2 model using the Trainer API.

Key Steps in train_model.py:
	1.	Data Preprocessing: Combines information into a single string of text.
	2.	Tokenization: Tokenizes the combined text into input-output pairs for training.
	3.	Model Training: Fine-tunes the GPT-2 model on the preprocessed data.

To train the model:

python Training_gpt2.py

 2. trained-model.py - Text Generation with LLaMA

This script demonstrates how to generate text using the meta-llama/Llama-3.2-1B-Instruct model, including how to create custom prompt templates for specific tasks. The script includes the following:
	•	Loads the LLaMA model and tokenizer.
	•	Sets up a chat-like interaction with system and user prompts.
	•	Generates responses based on these prompts.
	•	Fine-tunes the model using generated input-output pairs.

Key Steps in generate_text.py:
	1.	Prompt Setup: Defines a system and user prompt for a task.
	2.	Text Generation: Generates text using the pipeline API.
	3.	Fine-Tuning (Optional): Demonstrates how to prepare custom training data for fine-tuning the LLaMA model on specific tasks.

To run:

```python trained-model.py```

Model Details
	•	Pre-trained Models:
	•	meta-llama/Llama-3.2-1B-Instruct: Used for instruction-based text generation.
	•	openai-community/gpt2: Fine-tuned on the custom dataset.
	•	Device: MPS (Apple Silicon), but can be adjusted for other hardware configurations.

# Conclusion

This repo showcases my findings and work within the huggingface libraries as I navigate towards standing up my own model specfically fine-tuned for generating interatable HTML and JS elements 
