from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DefaultDataCollator
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

data_collator = DefaultDataCollator()
data_collator = DefaultDataCollator(return_tensors="pt")

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
dataset = load_dataset("math_dataset",'arithmetic__add_or_sub')['train'].shuffle(seed=42).select(range(5000))

def preprocess_data(examples):
    # Combine question and answer into one string
    # The model will learn to generate the answer following the question
    texts = [q + " Answer: " + a for q, a in zip(examples['question'], examples['answer'])]
    
    # Tokenize the texts. This will automatically add the necessary special tokens
    tokenized_inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=32)
    
    # For GPT-2 like models, the labels are usually the same as the input IDs
    # since the model is expected to predict the next token in the sequence
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    return tokenized_inputs

tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
print(tokenized_datasets)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
small_train_dataset = tokenized_datasets.select(range(1000))
small_eval_dataset =  tokenized_datasets.select(range(500))
print(small_train_dataset[0])
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
training_args = TrainingArguments(
    output_dir="./models/test",
    num_train_epochs=32,               # number of training epochs
    per_device_train_batch_size=4,    # batch size per device during training
    warmup_steps=10,                 # number of warmup steps
    weight_decay=0.01,                # strength of weight decay
    logging_dir="./logs",             # directory for storing logs
)

trainer = Trainer(
    model=model,
    args=training_args, 
    tokenizer=tokenizer,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()