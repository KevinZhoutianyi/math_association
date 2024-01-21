from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Trainer
import numpy as np
import evaluate
from transformers import DataCollatorForLanguageModeling
from datasets import load_metric
import os,re
import torch
import pdb



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
   
class Skipping_Transformer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # print(self.model.state_dict().keys())
    
    def forward(self, input_ids):
        print('my forward')
        outputs = self.model(input_ids,output_hidden_states=True)
        # print(outputs.hidden_states[0].shape)torch.Size([1, 7, 1600])
        # print(len(outputs.hidden_states))49
        # Extract the hidden states from the 20th layer
        representation = outputs.hidden_states[-1]  # 0-indexed, so 19 is the 20th layer

        # Use the last token's representation for prediction
        last_token_representation = representation[:, -1, :]
        

        # last_layernorm = self.model.state_dict()['transformer.ln_f.weight']
        # last_layernorm_bias = self.model.state_dict()['transformer.ln_f.bias']
        last_embeddingoutput = self.model.state_dict()['lm_head.weight']
        # Predict the next word using the last token representations
        # normalized_representation = torch.nn.functional.layer_norm(
        #     last_token_representation, 
        #     normalized_shape=last_token_representation.size()[1:], 
        #     weight=last_layernorm, 
        #     bias=last_layernorm_bias
        # )

    # Project to Embedding Space
        logits = torch.matmul(last_token_representation, last_embeddingoutput.t())

        # print(logits)tensor([[-2.7689, -0.6169, -1.6098,  ..., -6.0130, -5.5908,  4.3069]],
        # print(self.model(input_ids).logits[:,-1,:])tensor([[-2.7689, -0.6169, -1.6098,  ..., -6.0130, -5.5908,  4.3069]],
        return logits#torch.argmax(,dim=-1)


# Instantiate the modified model
path = '/project/vsharan_1180/Tianyi/rome/experiments/models/newdata/checkpoint-16480'
device = "cuda"


tokenizer = AutoTokenizer.from_pretrained(path)

# pdb.set_trace()
model = AutoModelForCausalLM.from_pretrained(path).to(device)
skip_finetuned = Skipping_Transformer(model).to(device)
# gptxl_mdoel = AutoModelForCausalLM.from_pretrained('gpt2-xl').to(device)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# skip_finetuned = Skipping_Transformer(model)





data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataset = load_dataset("math_dataset",'arithmetic__add_or_sub')['train'].shuffle(seed=42).select(range(500))



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
raw_input = ["3 12 +"]

for x in list(raw_input):
    inputs = tokenizer.encode(x, return_tensors="pt", max_length=100, truncation=True).to(device)
    pred_list = []
    print(inputs)
    for i in range(5):
        outputs = skip_finetuned(inputs)#.forward(inputs)
        pred_list.append(torch.argmax(outputs))
    print(pred_list)
    token_ids = [t.item() for t in pred_list]
    skip_finetuned_outputs = tokenizer.decode(token_ids)
    print('skip_finetuned_outputs',skip_finetuned_outputs)
    print('generate:',tokenizer.decode(model.generate(inputs,pad_token_id=tokenizer.eos_token_id,max_length= 60)[0]))
    print('skip_finetuned_outputs generate:',tokenizer.decode(skip_finetuned_outputs.generate(inputs,pad_token_id=tokenizer.eos_token_id,max_length= 60)[0]))
    # print(f"\n input:{x}  \n gptxl_output:{gptxl_outputs}\n output: {tokenizer.decode(outputs[0])} \n")
