# %% [markdown]
# <a href="https://colab.research.google.com/github/kmeng01/rome/blob/main/notebooks/rome.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" align="left"/></a>&nbsp;or in a local notebook.

# %%

# %%
IS_COLAB = False
ALL_DEPS = False
try:
    import google.colab, torch, os

    IS_COLAB = True
    os.chdir("/content/rome")
    if not torch.cuda.is_available():
        raise Exception("Change runtime type to include a GPU.")
except ModuleNotFoundError as _:
    pass

# %% [markdown]
# # Rank-One Model Editing (ROME)
# This notebook enables interactive experimentation with ROME and several other comparable baselines.
# The goal is to write new facts (e.g. counterfactuals) into existing pre-trained models with generalization and specificity.

# %%


# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_interactive, generate_fast

from experiments.py.demo import demo_model_editing, stop_execution

# %% [markdown]
# Here, you can specify a GPT model (`MODEL_NAME`).
# 
# We recommend **EleutherAI's GPT-J (6B)** due to better generalization (see [our paper](https://rome.baulab.info/) for details), but GPT-2 XL (1.5B) consumes less memory.
# * `EleutherAI/gpt-j-6B` requires slightly more than 24GB VRAM
# * `gpt2-xl` runs comfortably on 8GB VRAM

# %%
MODEL_NAME = "gpt2-xl"  # gpt2-{medium,large,xl} or EleutherAI/gpt-j-6B

# %%
model, tok = (
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, low_cpu_mem_usage=IS_COLAB).to(
        "cuda"
    ),
    AutoTokenizer.from_pretrained(MODEL_NAME),
)
tok.pad_token = tok.eos_token
print(model.config)

# %% [markdown]
# A requested rewrite can be specified using `request`. `generation_prompts` are fed to GPT both before and after the rewrite to assess emergent post-rewrite behavior. See the bottom of this notebook for more examples.
# 

# %%
request = [
    {
        "prompt": "{} was the founder of",
        "subject": "Steve Jobs",
        "target_new": {"str": "Microsoft"},
    }
]

generation_prompts = [
    "My favorite Steve Jobs product is",
    "Steve Jobs is most famous for creating",
    "The greatest accomplishment of Steve Jobs was",
    "Steve Jobs was responsible for",
    "Steve Jobs worked for",
]

# %% [markdown]
# This cell executes the model edit.
# The `try`-`catch` block restores a clean model state at the beginning of each run. `ALG_NAME` controls which algorithm is used. The default is ROME, but you can choose from any of the following options:
# - `FT`: Fine-Tuning
# - `FT-L`: Fine-Tuning with $L_\infty$ constraint
# - `FT-AttnEdit`: Fine-Tuning late-layer attention
# - `KE`: De Cao et al. Knowledge Editor
# - `KE-CF`: KE trained on CounterFact
# - `MEND`: Mitchell et al. Hypernetwork
# - `MEND-CF`: MEND trained on CounterFact
# - `MEND-zsRE`: MEND trained on zsRE QA
# - `ROME`: Our Rank-One Model Editing Method
# 
# Hyperparameters are refreshed from config files (located in `hparams/`) at each execution. To modify any parameter, edit and save the respective file. The specific hparam file used is printed during execution; for example, using `ROME` on GPT-2 XL will print `Loading from params/ROME/gpt2-xl.json`.
# 
# ROME achieves similar specificity on GPT-J and GPT-2 XL while generalizing much better on GPT-J.
# 

# %%
ALG_NAME = "ROME"

# %%
# Restore fresh copy of model
try:
    with torch.no_grad():
        for k, v in orig_weights.items():
            nethook.get_parameter(model, k)[...] = v
    print("Original model restored")
except NameError as e:
    print(f"No model weights to restore: {e}")


# Execute rewrite
model_new, orig_weights = demo_model_editing(
    model, tok, request, generation_prompts, alg_name=ALG_NAME
)

# %%
# stop_execution()

# %% [markdown]
# Use the cell below to interactively generate text with any prompt of your liking.

# %%
generate_interactive(model_new, tok, max_out_len=100, use_logit_lens=True)

# %% [markdown]
# Here are some extra request/prompt combinations you can try. Simply run them before the editing cell!

# %%
request = [
    {
        "prompt": "{} plays the sport of",
        "subject": "LeBron James",
        "target_new": {"str": "football"},
    }
]

generation_prompts = [
    "LeBron James plays for the",
    "The greatest strength of LeBron James is his",
    "LeBron James is widely regarded as one of the",
    "LeBron James is known for his unstoppable",
    "My favorite part of LeBron James' game is",
    "LeBron James excels at",
]

# %%
request = [
    {
        "prompt": "{} was developed by",
        "subject": "Mario Kart",
        "target_new": {
            "str": "Apple",
        },
    }
]

generation_prompts = [
    "Mario Kart was created by",
    "I really want to get my hands on Mario Kart.",
    "Mario Kart is",
    "Which company created Mario Kart?",
]

# %%



