import os
import re
import tiktoken
from icetk import icetk
import argparse
from tqdm import tqdm
from openai import OpenAI
from zhipuai import ZhipuAI


from config import (
    DATASET_MAXGEN, 
    DATASET_PROMPT, 
    DATASET_SELECTED, 
    DATASET_LENGTH_LEVEL,
)
from utils import (
    ensure_dir, 
    seed_everything,
    get_dataset_names,
    post_process,
    load_jsonl,
    load_LVEval_dataset,
    dump_preds_results_once,
)



def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    model_name,
    save_path,
    model_id,
):
    preds = []

    existed_questions = [data['input'] for data in load_jsonl(save_path)]

    for json_obj in tqdm(data):
        if json_obj['input'] in existed_questions:
            print(f'pred already exists in {save_path}, jump...')
            continue
        prompt = prompt_format.format(**json_obj)
        # following LongBench, we truncate to fit max_length
        tokenized_prompt = tokenizer.encode(prompt)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[:half]) + tokenizer.decode(tokenized_prompt[-half:])

        response = model.chat.completions.create(
            model = model_id,
            # do_sample = False,
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        pred = response.choices[0].message.content
        pred = post_process(pred, model_name)
        item = {
            "pred": pred,
            "answers": json_obj["answers"],
            "gold_ans": json_obj["answer_keywords"] if "answer_keywords" in json_obj else None,
            "input": json_obj["input"],
            "all_classes": json_obj["all_classes"] if "all_classes" in json_obj else None,
            "length": json_obj["length"],
        }
        dump_preds_results_once(item, save_path)
        preds.append(item)
    return preds

def single_processing(datasets, args):
    model_id = args.model_name
    if 'gpt' in model_id:
        client = OpenAI()
        tokenizer = tiktoken.encoding_for_model(model_id)
    elif 'glm' in model_id:
        client = ZhipuAI(api_key="************************")
        tokenizer = icetk


    for dataset in tqdm(datasets):
        datas = load_LVEval_dataset(dataset, args.data_path)
        dataset_name = re.split('_.{1,3}k', dataset)[0]
        prompt_format = DATASET_PROMPT[dataset_name]
        max_gen = DATASET_MAXGEN[dataset_name]
        save_path = os.path.join(args.output_dir, dataset + ".jsonl")
        preds = get_pred(
            client,
            tokenizer,
            datas,
            args.model_max_length,
            max_gen,
            prompt_format,
            args.model_name,
            save_path,
            model_id,
        )
        
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None, required=True, choices=["gpt-4-0613", "gpt-3.5-turbo-1106", "gpt-4-1106-preview", "glm-4", "glm-3-turbo"])
    parser.add_argument("--model-max-length", type=int, default=15500)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return process_args(parser.parse_args(args))

def process_args(args):
    return args

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    ensure_dir(args.output_dir)
    datasets = get_dataset_names(DATASET_SELECTED, DATASET_LENGTH_LEVEL)
    single_processing(datasets, args)
  