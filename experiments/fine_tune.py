from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorForLanguageModeling

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataset = load_dataset("math_dataset",'arithmetic__add_or_sub')['train'].shuffle(seed=42).select(range(500))

def preprocess_data(examples):
    # Combine question and answer into one string
    # The model will learn to generate the answer following the question
    texts = [q.lstrip("b'").replace("\\n'",'') + " Answer: " + a.lstrip("b'").replace("\\n'",'') for q, a in zip(examples['question'], examples['answer'])]
    
    # Tokenize the texts. This will automatically add the necessary special tokens
    tokenized_inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=32)
    
    # For GPT-2 like models, the labels are usually the same as the input IDs
    # since the model is expected to predict the next token in the sequence
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
print(tokenized_datasets)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
small_train_dataset = tokenized_datasets.select(range(400))
small_eval_dataset =  tokenized_datasets.select(range(400,500))


print(small_train_dataset[0])
print(data_collator([small_train_dataset[0],small_train_dataset[1]]))
print(tokenizer.decode(small_train_dataset[0]['input_ids']))
print(tokenizer.decode(small_train_dataset[0]['labels']))
training_args = TrainingArguments(
    output_dir="./models/test",
    num_train_epochs=32,               # number of training epochs
    per_device_train_batch_size=4,    # batch size per device during training
    warmup_steps=10,                 # number of warmup steps
    weight_decay=0.01,                # strength of weight decay
    gradient_accumulation_steps=16,
    logging_dir="./logs",             # directory for storing logs
    seed=123
)

trainer = Trainer(
    model=model,
    args=training_args, 
    tokenizer=tokenizer,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset
)

trainer.train()