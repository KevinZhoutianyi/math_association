
IS_COLAB = False

import sys
sys.path.append('..')


import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)

from experiments.causal_trace import (
    make_inputs,
    generate_sentence,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from dsets import KnownsDataset

torch.set_grad_enabled(False)

import warnings
warnings.filterwarnings("ignore")



def trace_with_patch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    trace_layers=None,  # List of traced outputs to return
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs

def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)

def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
        print(subject)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None,savepdf=None):
    for kind in [None, "mlp", "attn"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind,savepdf=savepdf+prompt.replace(" ", "_").replace(":","").replace("?","").replace(".",'')+'corrupt-'+subject+('state' if kind==None else kind)
        )


model_name =''#'finetune_gpt2_xl' #'gpt2-xl' #' #"gpt2-xl" #/project/vsharan_1180/Tianyi/rome/experiments/models/large/checkpoint-12432" #gpt2-xl"# 'EleutherAI_gpt-j-6B' #  # or "EleutherAI/gpt-j-6B" or "EleutherAI/gpt-neox-20b"
# folder_path = '/project/vsharan_1180/Tianyi/rome/experiments/models/llama'
# folder_names = [os.path.join(folder_path, name) for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
# print(folder_names)
folder_names =['/project/vsharan_1180/Tianyi/rome/math_exp/models/newdata2/checkpoint-24736']#['meta-llama/Llama-2-7b-hf']


for model_name in folder_names:
    model_name = model_name# ''
    mt = ModelAndTokenizer(
        model_name,
        low_cpu_mem_usage=IS_COLAB,
        torch_dtype=(torch.float16 if "20b" in model_name else None),
    )
    # if model_name=='meta-llama/Llama-2-7b-hf':
    #     mt.num_layers = mt.num_hidden_layers 

    print(predict_token(
        mt,
        ["3 12 + ", "3 12 + 1"],
        return_p=True,
    ))
    print(predict_token(
        mt,
        ["What is 3 plus 12? Answer: ", "What is 3 plus 12? Answer: 1"],
        return_p=True,
    ))
    # print('generate:',generate_sentence(
    #     mt,
    #     ["3 12 + ", "What is 3 plus 12? Answer: "],
    # ))
    # print('---')

    knowns = KnownsDataset(DATA_DIR)  # Dataset of known facts
    noise_level = 3 * collect_embedding_std(mt, [k["subject"] for k in knowns])
    print(f"Using noise level {noise_level}")

    print('plot_all_flow')
    # plot_all_flow(mt, "3  12  +  ", '3',noise=noise_level, savepdf='/project/vsharan_1180/Tianyi/rome/math_exp/casual_tracing_exp/'+model_name.rsplit('-', 1)[-1]+'/' )
    # plot_all_flow(mt, "What is 3 plus 12? Answer: ", '3',noise=noise_level, savepdf='/project/vsharan_1180/Tianyi/rome/math_exp/casual_tracing_exp/'+model_name.rsplit('-', 1)[-1]+'/' )
    # plot_all_flow(mt, "3  12  +  ", '12',noise=noise_level, savepdf='/project/vsharan_1180/Tianyi/rome/llama/my_exp_res1/'+model_name.rsplit('-', 1)[-1]+'/' )
    # plot_all_flow(mt, "3  12  +  ", '+',noise=noise_level, savepdf='/project/vsharan_1180/Tianyi/rome/llama/my_exp_res2/'+model_name.rsplit('-', 1)[-1]+'/' )
    plot_all_flow(mt, "The Space Needle is in downtown", 'The Space Needle',noise=noise_level, savepdf='/project/vsharan_1180/Tianyi/rome/math_exp/casual_tracing_exp/'+model_name.rsplit('-', 1)[-1]+'/' )
   