import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution
IS_COLAB = False
MODEL_NAME = 'gpt2-xl' #"gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
print(model)