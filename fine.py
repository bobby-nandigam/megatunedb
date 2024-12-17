from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load the pre-trained GPT-2 model and tokenizer
model_name = "rakeshkiriyath/gpt2Medium_text_to_sql"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset
def preprocess_data(examples):
    inputs = [f"Translate the following English question to SQL: {question}" for question in examples['question']]
    outputs = examples['sql']
    
    # Tokenize inputs and outputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=512, truncation=True, padding="max_length")["input_ids"]

    # Add a special token for ignored positions in the loss
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels
    ]

    # Debugging: Print the shapes of inputs and labels
    print("Input IDs shape:", len(model_inputs["input_ids"][0]))  # Print shape of the first input
    print("Labels shape:", len(labels[0]))  # Print shape of the first label

    # Update model_inputs with labels
    model_inputs["labels"] = labels
    return model_inputs

# Load your dataset (assuming it's in CSV files)
dataset = load_dataset("csv", data_files={"train": "train.csv", "validation": "validation.csv"})

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_data, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-sql",
    evaluation_strategy="steps",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    save_steps=500,
    logging_dir="./logs",
    logging_steps=100,
    push_to_hub=False,
    fp16=False,  # Ensure fp16 is disabled for CPU
    no_cuda=True,  # Explicitly disable GPU usage
)

# Custom Trainer to override compute_loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Debugging: Print shapes during training
        print("Logits shape:", logits.shape)
        print("Labels shape:", labels.shape)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Create an instance of CustomTrainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./gpt2-finetuned-sql")
tokenizer.save_pretrained("./gpt2-finetuned-sql")
print("Model fine-tuned and saved successfully!")
"""
# Sample queries related to log data
queryList = [
    "Display errors from the last 3 months.",
    "Display errors from the last 1 month.",
    "Display log_data from the last 1 hour.",
    "Count error logs for the last 7 days.",
    "List distinct error messages and their count in the last 30 days.'",
    "Find the first occurrence of each log level.",
    "Count log_data by day for the last week.",
    "Display log_data generated during a specific time range.",
    "Display all log_data grouped by log level.",
    "Count the number of log_data in each month of the current year.",
    "Find the maximum gap between consecutive log_data."
]

# Process the queries to generate SQL
for query in queryList:
    sql_result = generate_text_to_sql(query, finetunedGPT, finetunedTokenizer)
    print(f"Query: {query}\nGenerated SQL: {sql_result}\n")

"""