from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")
# Function to check if a number is tokenized as a single token
def is_single_token(num):
    text = str(num)
    tokenized_text = tokenizer.tokenize(text)
    return len(tokenized_text) == 1

# Finding the maximum single-token number
max_num = 0
for num in range(1, 1000000):  # You might need to adjust this range
    if is_single_token(num):
        max_num = num
    else:
        break

print(f"The maximum number tokenized as a single token is: {max_num}")