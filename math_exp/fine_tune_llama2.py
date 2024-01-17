from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorForLanguageModeling
from datasets import load_metric
import os,re
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from decimal import Decimal, getcontext

# Set the precision (number of significant digits)
getcontext().prec = 32

metric = evaluate.load('evaluate-metric/bleu')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
   
    predictions = tokenizer.batch_decode(predictions,skip_special_tokens =True)
    labels = tokenizer.batch_decode(labels,skip_special_tokens =True)
    met = metric.compute(predictions=predictions, references=labels)
    met['precisions'] = sum(met['precisions']) / len(met['precisions'])
    return met

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
datasize = 100000
evalsize = 1000
dataset = load_dataset("math_dataset",'arithmetic__add_or_sub')['train'].shuffle(seed=42).select(range(datasize)) #2M is the total size, now we use 200000

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
    # Remove unwanted characters from question and answer



    # # make the data in format from 'What is 0.4 minus 111847786743.475? to '0.4, 111847786743.475, -'
    # preprocessed_data = []

    # for question, answer in zip(examples['question'], examples['answer']):
    #     # Remove unwanted characters from question and answer
    #     question = question.lstrip("b'").replace("\\n'",'')
    #     answer = answer.lstrip("b'").replace("\\n'",'').strip()

    #     # Extract operands from the question using regex
    #     operands = re.findall(r'-?\d+(?:\.\d+)?', question)

    #     if len(operands) == 2:
    #         operand1, operand2 = operands
    #         operand1 = Decimal(operand1)
    #         operand2 = Decimal(operand2)
    #         answer = Decimal(answer)

    #         # Determine if the operation is sum or difference
    #         if abs(operand1 + operand2 - answer) < 1:  # Consider floating point precision
    #             operation = '+'
    #             formatted_data = f"{operand1}  {operand2}  {operation}  {answer}"
    #         elif abs(operand1 - operand2 - answer) < 1:
    #             operation = '-'
    #             formatted_data = f"{operand1}  {operand2}  {operation}  {answer}"
    #         elif abs(operand2 - operand1 - answer) < 1:
    #             operation = '-'
    #             formatted_data = f"{operand2}  {operand1}  {operation}  {answer}"
    #         else:
    #             print(question,answer)
    #             operands = re.findall(r'-?\d+(?:\.\d+)?', question)
    #             print(operands)
    #             continue  # Skip if neither

    #         preprocessed_data.append(formatted_data)
            
    # tokenized_inputs = tokenizer(preprocessed_data, truncation=True, padding='max_length', max_length=32)

    # # For GPT-2 like models, the labels are usually the same as the input IDs
    # # since the model is expected to predict the next token in the sequence
    # tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    # return tokenized_inputs



tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
print(tokenized_datasets)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
small_train_dataset = tokenized_datasets.select(range(datasize-evalsize))
small_eval_dataset =  tokenized_datasets.select(range(datasize-evalsize,datasize))


print(small_train_dataset[0])
print(data_collator([small_train_dataset[0],small_train_dataset[1]]))
print(tokenizer.decode(small_train_dataset[0]['input_ids']))
print(tokenizer.decode(small_train_dataset[0]['labels']))
training_args = Seq2SeqTrainingArguments(
    output_dir="./models/newdata",
    num_train_epochs=32,               # number of training epochs
    per_device_train_batch_size=1,    # batch size per device during training
    warmup_steps=10,                 # number of warmup steps
    weight_decay=0.01,                # strength of weight decay
    gradient_accumulation_steps=16,
    logging_dir="./logs",             # directory for storing logs
    seed=1,
    logging_steps = 32,
    save_strategy ="epoch",
    evaluation_strategy = 'steps',
    save_total_limit = 200,
    do_predict = True,
    # load_best_model_at_end  = True,
    
)

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./models/newdata",
#     num_train_epochs=32,               # number of training epochs
#     per_device_train_batch_size=12,    # batch size per device during training
#     warmup_steps=10,                 # number of warmup steps
#     weight_decay=0.01,                # strength of weight decay
#     gradient_accumulation_steps=16,
#     logging_dir="./logs",             # directory for storing logs
#     seed=1,
#     logging_steps = 32,
#     save_strategy ="steps",
#     save_steps = 6,
#     evaluation_strategy = 'steps',
#     save_total_limit = 200,
#     do_predict = True,
#     # load_best_model_at_end  = True,
    
# )

trainer = Trainer(
    model=model,
    args=training_args, 
    tokenizer=tokenizer,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()