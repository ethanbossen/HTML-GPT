from transformers import AutoTokenizer, AdamW
from transformers import AutoModelForCausalLM
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-1B-Instruct"
device = "mps"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16,
                                             device_map=device)

generation_pipeline = pipeline(task="text-generation",
                               model=model, tokenizer=tokenizer)


prompt_template = [
    {
        "role": "system",
        "content": "You are an AI assistant that generates html and js code"
    },
    {
        "role": "user",
        "content": "generate a simple pop-up overlay widget"
    },
    {
        "role": "assistant",
        "content": "<div>"
    }
]

tokenizer.pad_token = tokenizer.eos_token

tokenized = tokenizer.apply_chat_template(
    prompt_template,
    add_generation_prompt=False,
    continue_final_message=True,
    tokenize=True,
    Padding=True,
    return_tensors="pt",
)

out = model.generate(tokenized.to("mps"), max_new_tokens=56)
# - will only run if tokenize is set to true in tokenized

decoded = tokenizer.batch_decode(out)

# Print the decoded text
print(decoded[0])

training_prompt = [
    {
        "role": "user", "content": "What's the best pizza in Grand Rapids"
    },
    {
        "role": "assistant", "content": "The best pizza in Grand Rapids is"
    }
]
target_response = "Russo's"

test_tokenized = tokenizer.apply_chat_template(training_prompt, continue_final_message=True, return_tensors="pt").to(device=device)
test_out = model.generate(test_tokenized.to("mps"), max_new_tokens=20)
print(tokenizer.batch_decode(test_out))

from transformers import AdamW


def generate_input_output_pair(prompt, target_responses):
    # Apply the chat template
    chat_templates = tokenizer.apply_chat_template(prompt, continue_final_message=True, tokenize=False)

    # Combine the chat templates with target responses
    full_response_text = [
        (chat_template + " " + target_response + tokenizer.eos_token)
        for chat_template, target_response in zip(chat_templates, target_responses)
    ]

    # Tokenize the full response text
    input_ids_tokenized = tokenizer(
        full_response_text,
        return_tensors="pt",
        max_length=64,  # Keep sequence length manageable
        truncation=True,
        padding="max_length"  # Ensure consistent sizes
    )["input_ids"]

    # Tokenize the labels (responses)
    labels_tokenized = tokenizer(
        [" " + response + tokenizer.eos_token for response in target_responses],
        add_special_tokens=False, return_tensors="pt", padding="max_length", max_length=input_ids_tokenized.shape[1]
    )["input_ids"]

    # Fix the labels tokenized (set padding tokens to -100)
    labels_tokenized_fixed = torch.where(labels_tokenized != tokenizer.pad_token_id, labels_tokenized, -100)

    # Shift input_ids and labels for causal language modeling
    input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
    labels_tokenized_right_shifted = labels_tokenized_fixed[:, 1:]

    # Create attention mask
    attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id

    return {
        "input_ids": input_ids_tokenized_left_shifted,
        "attention_mask": attention_mask,
        "labels": labels_tokenized_right_shifted,
    }

model.config.num_hidden_layers = 6  # Reduce layers

data = generate_input_output_pair(
    prompt = [
        {"role": "user", "content": "What's the best pizza in Grand Rapids"},
        {"role": "assistant", "content": "Best Pizza:"},
    ],
    target_responses = ["Russo's"]
)
data["input_ids"] = data["input_ids"].to(device=device)
data["labels"] = data["labels"].to(device=device)

optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

import torch.nn as nn

# def calculate_loss(logits, labels):
#     loss_fn = nn.CrossEntropyLoss(reduction="none")
#     cross_entrophy_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
#     return cross_entrophy_loss
#
# scaler = torch.cuda.amp.GradScaler()  # Use scaler for mixed precision training
#
# for _ in range(10):
#     with torch.cuda.amp.autocast():  # Automatic casting for FP16
#         out = model(input_ids=data["input_ids"].to(device))
#         loss = calculate_loss(out.logits, data["labels"]).mean()
#
#     scaler.scale(loss).backward()
#     scaler.step(optimizer)
#     scaler.update()
#     optimizer.zero_grad()
#
#     print("loss: ", loss.item())