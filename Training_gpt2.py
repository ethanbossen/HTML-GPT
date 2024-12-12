from transformers import pipeline, AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer
from datasets import load_dataset

pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")

# ans = pipe("How are you?")

dataset = load_dataset("Jarbas/metal-archives-bands")

def preprocess_function(examples):
    name = examples["name"] or "Unknown Name"
    genre = examples["genre"] or "Unknown Genre"
    theme = examples["theme"] or "Unknown Theme"

    combined_text = (
            "Band Name: " + name + "\n"
            "Genre: " + genre + "\n"
            "Theme: " + theme
    )
    return {"text": combined_text}

processed_dataset = dataset.map(preprocess_function, batched=False)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], padding=True, truncation=True)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = processed_dataset.map(tokenize_function, batched=False)

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()