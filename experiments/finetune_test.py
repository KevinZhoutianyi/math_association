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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


path = '/project/vsharan_1180/Tianyi/rome/experiments/models/newdata/checkpoint-16480'
device = "cuda"


tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path).to(device)
gptxl_mdoel = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataset = load_dataset("math_dataset",'arithmetic__add_or_sub')['train'].shuffle(seed=42).select(range(500))
def preprocess_data(examples):
    # # Combine question and answer into one string
    # # The model will learn to generate the answer following the question
    # texts = [q.lstrip("b'").replace("\\n'",'') + " Answer: " + a.lstrip("b'").replace("\\n'",'') for q, a in zip(examples['question'], examples['answer'])]
    
    # # Tokenize the texts. This will automatically add the necessary special tokens
    # tokenized_inputs = tokenizer(texts, truncation=True, padding='max_length', max_length=32)
    
    # # For GPT-2 like models, the labels are usually the same as the input IDs
    # # since the model is expected to predict the next token in the sequence
    # tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    
    # return tokenized_inputs
    preprocessed_data = []

    for question, answer in zip(examples['question'], examples['answer']):
        # Remove unwanted characters from question and answer
        question = question.lstrip("b'").replace("\\n'",'')
        answer = answer.lstrip("b'").replace("\\n'",'').strip()

        # Extract operands from the question using regex
        operands = re.findall(r'-?\d+(?:\.\d+)?', question)

        if len(operands) == 2:
            operand1, operand2 = operands
            operand1 = Decimal(operand1)
            operand2 = Decimal(operand2)
            answer = Decimal(answer)

            # Determine if the operation is sum or difference
            if abs(operand1 + operand2 - answer) < 1:  # Consider floating point precision
                operation = '+'
                formatted_data = f"{operand1}  {operand2}  {operation}  {answer}"
            elif abs(operand1 - operand2 - answer) < 1:
                operation = '-'
                formatted_data = f"{operand1}  {operand2}  {operation}  {answer}"
            elif abs(operand2 - operand1 - answer) < 1:
                operation = '-'
                formatted_data = f"{operand2}  {operand1}  {operation}  {answer}"
            else:
                print(question,answer)
                operands = re.findall(r'-?\d+(?:\.\d+)?', question)
                print(operands)
                continue  # Skip if neither

            preprocessed_data.append(formatted_data)
            
    tokenized_inputs = tokenizer(preprocessed_data, truncation=True, padding='max_length', max_length=32)

    # For GPT-2 like models, the labels are usually the same as the input IDs
    # since the model is expected to predict the next token in the sequence
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs


tokenized_datasets = dataset.map(preprocess_data, batched=True, remove_columns=dataset.column_names)
print(tokenized_datasets)
# tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
small_train_dataset = tokenized_datasets.select(range(400))
small_eval_dataset =  tokenized_datasets.select(range(400,500))


# print(small_train_dataset[0])
# print(data_collator([small_train_dataset[0],small_train_dataset[1]]))
print(f"training data sample: {tokenizer.decode(small_train_dataset[0]['input_ids'])}")

# print(tokenizer.decode(small_train_dataset[0]['labels']))



print('\n Test on some unseen data')
raw_input = ["3  4  +  ", "5  1  -  ", "159406555028082  -0.7  -", "3  12  +"]

for x in list(raw_input):
    inputs = tokenizer.encode(x, return_tensors="pt", max_length=100, truncation=True).to(device)
    outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id,max_length= 60)
    gptxl_outputs = tokenizer.decode(gptxl_mdoel.generate(inputs, pad_token_id=tokenizer.eos_token_id,max_length= 60)[0]).replace('\n','')
    print(f"\n input:{x}  \n gptxl_output:{gptxl_outputs}\n output: {tokenizer.decode(outputs[0])} \n")