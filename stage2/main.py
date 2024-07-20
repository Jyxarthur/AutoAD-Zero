import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
# modify the path below
os.environ['TRANSFORMERS_CACHE'] = "/path/to/cache"
import sys
import torch
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from promptloader import get_user_prompt

def initialise_model(access_token):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=access_token,
    )
    return pipeline

def summary_each(pipeline, user_prompt):    
    sys_prompt = (
            "[INST] <<SYS>>\nYou are an intelligent chatbot designed for summarizing TV series audio descriptions. "
            "Here's how you can accomplish the task:------##INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
            "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end. \n<</SYS>>\n\n{} [/INST] "
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.6,
        top_p=0.9,
    )

    output_text = outputs[0]["generated_text"][len(prompt):]
    return output_text

def main(args):
     # initialise the model
    pipeline = initialise_model(args.access_token)
   
    # read predicted output from stage 1
    pred_df = pd.read_csv(args.pred_path)

    # apply slightly different prompt for movie and TV series
    if args.dataset == "tvad":
        if args.prompt_idx is None:
            args.prompt_idx = 0
        verb_list = ['look', 'walk', 'turn', 'stare', 'take', 'hold', 'smile', 'leave', 'pull', 'watch', 'open', 'go', 'step', 'get', 'enter']
    elif args.dataset in ["cmdad", "madeval"]:
        if args.prompt_idx is None:
            args.prompt_idx = 1
        verb_list = ['look', 'turn', 'take', 'hold', 'pull', 'walk', 'run', 'watch', 'stare', 'grab', 'fall', 'get', 'go', 'open', 'smile']
    else:
        print("Check the dataset name")
        sys.exit()


    text_gen_list = []
    text_gt_list = []
    start_sec_list = []
    end_sec_list = []
    vid_list = []
    for _, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        text_gt = row['text_gt']
        text_pred = row['text_gen']
        # complete the user_prompt
        user_prompt = get_user_prompt(prompt_idx=args.prompt_idx, verb_list=verb_list, text_pred=text_pred, duration=round(row['end_']-row['start_'], 2))

        # get ad summary for each prediction
        text_summary = summary_each(pipeline, user_prompt)
        text_summary = text_summary.replace("{'summarised_AD': '", "").replace("'}", "")
        
        # post-cleaning, output "" (empty) if the summary format is wrong
        if "1. Main characters:" in text_summary:
            text_summary = ""
        if "{" in text_summary or "}" in text_summary:
            text_summary = text_summary.replace("}", "").replace("{", "")

        text_gen_list.append(text_summary)
        text_gt_list.append(text_gt)
        start_sec_list.append(row['start'])
        end_sec_list.append(row['end'])
        vid_list.append(row['vid'])

    output_df = pd.DataFrame.from_records({'vid': vid_list, 'start': start_sec_list, 'end': end_sec_list, 'text_gt': text_gt_list, 'text_gen': text_gen_list})
    output_df.to_csv(os.path.join(os.path.dirname(args.pred_path), f"stage2_llama3_{args.prompt_idx}_" + os.path.basename(args.pred_path)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--access_token', default=None, type=str, help='HuggingFace token to access llama3')
    parser.add_argument('--prompt_idx', default=None, type=int, help='optional, use to indicate you own prompt')
    args = parser.parse_args()

    if args.access_token is None:
        print("Please access LLaMA3 on https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and add the access token.")
        sys.exit()

    main(args)
   