import os
import ast
import sys
import torch
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from promptloader import get_user_prompt
from openai import OpenAI
os.environ["OPENAI_API_KEY"] = #<insert-your-api-key> 


def summary_each(client, user_prompt, dataset):      
    if 'cmd' in dataset or 'mad' in dataset:
        dataset_text = "movie"
    elif 'tv' in dataset:
        dataset_text = "TV series"

    sys_prompt = (
            f"You are an intelligent chatbot designed for summarizing {dataset_text} audio descriptions. "
            "Here's how you can accomplish the task:------##INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
            "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end."
    )
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt  
        },
        {
            "role": "user",
            "content": user_prompt  
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )

    output_text = response.choices[0].message.content
    return output_text


def main(args):
    # read predicted output from stage 1
    pred_df = pd.read_csv(args.pred_path)

    # initialise openai client
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
    
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
        text_summary = summary_each(client, user_prompt, args.dataset)
        try:
            text_summary = ast.literal_eval(text_summary)['summarised_AD']
        except:
            text_summary = ""


        text_gen_list.append(text_summary)
        text_gt_list.append(text_gt)
        start_sec_list.append(row['start'])
        end_sec_list.append(row['end'])
        vid_list.append(row['vid'])

    output_df = pd.DataFrame.from_records({'vid': vid_list, 'start': start_sec_list, 'end': end_sec_list, 'text_gt': text_gt_list, 'text_gen': text_gen_list})
    output_df.to_csv(os.path.join(os.path.dirname(args.pred_path), f"stage2_gpt_{args.prompt_idx}_" + os.path.basename(args.pred_path)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--prompt_idx', default=None, type=int, help='optional, use to indicate you own prompt')
    args = parser.parse_args()

    main(args)
   