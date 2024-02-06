import os
import json
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)



def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_dataset_names(dataset_names, length_levels):
    datasets = []
    for name in dataset_names:
        for length in length_levels:
            datasets.append(f"{name}_{length}")

    return datasets

def dump_preds_results_once(pred, save_path):
    with open(save_path, "a+", encoding="utf-8") as f:
        json.dump(pred, f, ensure_ascii=False)
        f.write("\n")

def dump_preds_results(preds, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for pred in preds:
            json.dump(pred, f, ensure_ascii=False)
            f.write("\n")
        print(f"results saving >>>>>>>>> {save_path}")

def load_jsonl(data_path):
    datas = []
    if os.path.exists(data_path):
        f = open(data_path, 'r')
        for line in f.readlines():
            datas.append(json.loads(line))
    else:
        print(f"not exists: {data_path}")
    return datas

def load_LVEval_dataset(dataset_name, data_path=None):
    print(f"loading dataset >>>>>>>>> {dataset_name}")
    if data_path: # load from local path
        datas = []
        data_path = os.path.join(data_path, dataset_name) + ".jsonl"
        datas = load_jsonl(data_path)
        print(f"dataset path >>>>>>>>> {data_path}")
    else: # load from huggingface
        datas = load_dataset("infini-ai/LVEval", dataset_name, split='test', token=True)
    return list(datas)

def load_model_and_tokenizer(model_path, device):
    print(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attention_2=True, 
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.bfloat16, use_flash_attention_2=False, 
        )

    model = model.eval()
    return model, tokenizer

def load_model_and_tokenizer_once(id, model_path, device_dict=None, lock=None):
    device = torch.device(f"cuda:{id}") if id != -1 else "auto"
    print(f"using device {device}")
    model, tokenizer = load_model_and_tokenizer(
        model_path, device
    )
    if device_dict is None:
        return model, tokenizer
    if lock:
        with lock:
            device_dict[id] = (model, tokenizer)
    else:
        device_dict[id] = (model, tokenizer)

def model_generate(tokenizer, prompt, max_gen, model):
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
    context_length = input.input_ids.shape[-1]
    output = model.generate(
        **input,
        max_new_tokens=max_gen,
        do_sample=False,
    )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred

def truncate_prompt(tokenizer, prompt, max_length):
    # following LongBench, we truncate middle content to fit max_length
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > max_length:
        half = int(max_length / 2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    return prompt

def build_chat(tokenizer, prompt, model_name):
    if "chatglm2" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "BlueLM" in model_name:
        prompt = f"[|Human|]:{prompt}[|AI|]:"
    elif "vicuna" in model_name or "sft" in model_name:
        system_message = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        prompt = f"{system_message} USER: {prompt} ASSISTANT:"
    elif "llama2" in model_name or "Llama-2" in model_name or "LLaMA" in model_name:
        prompt = f"[INST]{prompt}[/INST]\n\n"
    elif "Mistral" in model_name:
        prompt = f"<s>[INST] {prompt} [/INST]"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    else:
        prompt = f"{prompt}"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
        
    return response