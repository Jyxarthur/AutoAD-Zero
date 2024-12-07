import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

from promptloader import get_general_prompt
from dataloader import MADEval_FrameLoader, TVAD_FrameLoader, CMDAD_FrameLoader

from openai import OpenAI
os.environ["OPENAI_API_KEY"] = # <insert-your-api-key>

def main(args):
    # initialize openai client
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
   
    # formulate text prompt template
    general_prompt = get_general_prompt(args.prompt_idx)
    
    # build dataloader
    if args.dataset == "tvad":
        D = TVAD_FrameLoader
        video_type = "TV series"
    elif args.dataset == "cmdad":
        D = CMDAD_FrameLoader
        video_type = "movie"
    elif args.dataset == "madeval":
        D = MADEval_FrameLoader
        video_type = "movie"
    else:
        print("Check dataset name")
        sys.exit()

    ad_dataset = D(tokenizer=None, processor=None, general_prompt=general_prompt, video_type = video_type,
                                anno_path=args.anno_path, charbank_path=args.charbank_path, video_dir=args.video_dir,
                                label_type=args.label_type, label_width=args.label_width, label_alpha=args.label_alpha)

    # Only support batch size = 1
    assert args.batch_size == 1
    loader = torch.utils.data.DataLoader(ad_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                            collate_fn=ad_dataset.collate_fn, shuffle=False, pin_memory=True)

    start_sec = []
    end_sec = []
    start_sec_ = []
    end_sec_ = []
    text_gt = []
    text_gen = []
    vids = []
    for idx, input_data in tqdm(enumerate(loader), total=len(loader), desc='EVAL'): 
        video_inputs = input_data["video"][0]
        texts = input_data["prompt"][0]
        messages = [
            {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": texts,
                        },
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "low"
                                },
                            }
                            for base64_image in video_inputs
                        ],
                    ],
                }
            ]
        # Specify your model here
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
        )

        output_text = response.choices[0].message.content
        outputs = [output_text.replace("###ANSWER TEMPLATE###:", "").strip()]
        vids.extend(input_data["imdbid"])
        text_gt.extend(input_data["gt_text"])
        text_gen.extend(outputs) 
        start_sec.extend(input_data["start"])
        end_sec.extend(input_data["end"])
        start_sec_.extend(input_data["start_"])
        end_sec_.extend(input_data["end_"])
        
    output_df = pd.DataFrame.from_records({'vid': vids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
    os.makedirs(f"{args.output_dir}/{args.dataset}_ads", exist_ok=True)
    print(f"Save csv to {args.output_dir}/{args.dataset}_ads")
    output_df.to_csv(f'{args.output_dir}/{args.dataset}_ads/stage1_gpt_{args.save_prefix}-{args.label_type}-{args.prompt_idx}.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('llama')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', default="tvad", type=str)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--save_prefix', default="", type=str)
    parser.add_argument('--model_path', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--charbank_path', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--video_dir', default=None, type=str)
    parser.add_argument('--label_type', default="circles", type=str)
    parser.add_argument('--label_width', default=10, type=int, help='label_width, 10 in a canvas 1000')
    parser.add_argument('--label_alpha', default=0.8, type=float)
    parser.add_argument('-j', '--num_workers', default=8, type=int, help='init mode')
    parser.add_argument('--max_exp', default=5, type=int, help='maximum number of repeating experiments')
    args = parser.parse_args()

    main(args)

    
    