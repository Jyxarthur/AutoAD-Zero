import os
import sys
import ast
import copy
import json
import torch
import pickle
import tarfile
import numpy as np
import pandas as pd

from PIL import Image
from utils import process_video, expand2square, read_tarfile, convert_bounding_box_to_rectangle, convert_bounding_box_to_ellipse, text_to_token


class MADEval_FrameLoader():
    def __init__(self,
                tokenizer,
                processor,
                general_prompt,
                video_type,
                label_type, 
                label_width, 
                label_alpha,
                anno_path,
                video_dir,
                charbank_path,
                **kwargs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.general_prompt=general_prompt
        self.video_type = video_type
        self.processor_mean = self.processor.image_mean

        # label information, including colour coding, type, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha

        # load annotation file
        self.anno_df = pd.read_csv(anno_path)
        
        # prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        charbank_pred = pickle.load(open(charbank_path,'rb'))
        charbank_pred = {str(key):value for key,value in charbank_pred.items()}
        for movie_name in charbank_pred.keys():
            charbank_pred_per_frame = charbank_pred[movie_name]
            name_ids_per_frame = charbank_pred_per_frame["charbank_nmids"].iloc[0]
            roles_per_frame = charbank_pred_per_frame["charbank_roles"].iloc[0]
            self.charbank_dict[movie_name] = {k: v for k, v in zip(name_ids_per_frame, roles_per_frame)}
  
        self.all_clips = []
        for anno_idx, anno_row in self.anno_df.iterrows():
            movie_name = anno_row['movie']
            imdbid = anno_row['imdb']
            lsmdc_filename = anno_row['lsmdc_filename']
            sentence = anno_row['sentence']
            start = 0
            end = anno_row['end'] - anno_row['start']
            video_path = os.path.join(video_dir, movie_name, lsmdc_filename + '.avi')
            if os.path.exists(video_path):
                self.all_clips.append((imdbid, movie_name, video_path, start, end, anno_row["start"], anno_row["end"], sentence, anno_row["bboxes"], anno_row["pred_ids"]))
        print(f"In total {len(self.all_clips)} MAD-Eval clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, movie_name, video_path, start, end, start_, end_, gt_text, bboxes, name_ids = self.all_clips[index]
        
        # load 8 frames 
        frames = process_video(video_path, num_frames=8, start=start, end=end)
        
        # since the face bboxes and corresponding name ids are predicted for 16 frames, 
        # below takes the results from frame 0, 2, 4, ..., to form 8-frame results.
        bboxes = ast.literal_eval(bboxes)
        bboxes = bboxes[::2]
        name_ids = ast.literal_eval(name_ids)    
        bboxes_filtered = []   # only bboxes corresponding to a recognised character is kept
        all_name_ids = {}  # in the form of {name_id_1: 1, name_id_2: 2, ...}
        for frame_idx in range(len(frames)):
            bboxes_filtered_per_frame = {}
            for name_idx, (name_id, bbox_idx_list) in enumerate(name_ids.items()):
                for bbox_idx in bbox_idx_list:
                    if bbox_idx[0] == int(frame_idx * 2) and name_id not in bboxes_filtered_per_frame.keys():
                        bboxes_filtered_per_frame[name_id] = bboxes[frame_idx][bbox_idx[1]]
                        if name_id not in all_name_ids.keys():
                            all_name_ids[name_id] = len(all_name_ids)
            bboxes_filtered.append(bboxes_filtered_per_frame)

        if self.label_type=="none":
            processed_frames = [expand2square(frame, tuple(int(x*255) for x in self.processor_mean)) for frame in frames]
        else:
            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: # skip as no bbox presents in this frame
                    processed_frames.append(expand2square(frame, tuple(int(x*255) for x in self.processor_mean)))
                    continue
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()

                        # overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))
                    processed_frame = expand2square(processed_frame, tuple(int(x*255) for x in self.processor_mean)) 
                    processed_frames.append(processed_frame)
        
        video_tensor = self.processor.preprocess(processed_frames, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, non_blocking=True)

        # Uncomment for visualisations
        # for frame_idx, processed_frame in enumerate(processed_frames):
        #     os.makedirs("tmp", exist_ok = True)
        #     filename = f"tmp/{imdbid}-{index}-{frame_idx}.jpg"
        #     processed_frame.save(filename)
        #     print(f"---Save file {filename} ---")
    
        # formulate the character name text prompt
        charbank_dict = self.charbank_dict[movie_name]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  #.split(" ")[0]
        
        if char_text == ". Possible characters (labeled by {label_type}): ": # no character recognised
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text="", label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)
        else:
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text= char_text.format(label_type=self.label_type), label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)

        return_dict =  {'video': video_tensor, \
                'imdbid': imdbid, \
                'input_id': input_id,
                'prompt': text_prompt,
                'gt_text': gt_text,
                'start': start, 
                'end': end, 
                'start_': start_, 
                'end_': end_, 
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['input_id'] = [sample['input_id'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        return out_batch



class TVAD_FrameLoader():
    def __init__(self,
                tokenizer,
                processor,
                general_prompt,
                video_type,
                label_type, 
                label_width, 
                label_alpha,
                anno_path,
                video_dir,
                charbank_path,
                **kwargs):

        self.processor = processor
        self.tokenizer = tokenizer
        self.general_prompt=general_prompt
        self.video_type = video_type
        self.processor_mean = self.processor.image_mean
        
        # label information, including colour coding, type, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha

        # load annotation file
        self.anno_df = pd.read_csv(anno_path)
    
        # prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        with open(os.path.join(charbank_path)) as fobj:
            charbank_dict = json.load(fobj)
        for key in charbank_dict.keys():
            self.charbank_dict[key] = {single_charbank_dict["id"]:single_charbank_dict["role"] for single_charbank_dict in charbank_dict[key]}         

        self.all_clips = []
        for anno_idx, anno_row in self.anno_df.iterrows():
            seg_name = anno_row["tvad_name"]
            if "friends" in seg_name:
                seg_path = os.path.join(video_dir, "friends_frames", seg_name + ".tar")
            else:
                seg_path = os.path.join(video_dir, "bbt_frames", seg_name + ".tar")
            if os.path.exists(seg_path):
                self.all_clips.append((anno_row["imdbid_pe"], seg_path, anno_row["tvad_index"], anno_row["scaled_start"], anno_row["scaled_end"], anno_row["start"], anno_row["end"], anno_row["text"], anno_row["bboxes"], anno_row["pred_ids"]))
        print(f"In total {len(self.all_clips)} TV-AD clips")


    def __len__(self):
        return len(self.all_clips)


    def __getitem__(self, index):
        imdbid, seg_path, tvad_indices, start, end, start_, end_, gt_text, bboxes, name_ids = self.all_clips[index]

        # load 8 frames
        frames = []
        for tvad_idx in ast.literal_eval(tvad_indices)[::2]: # since 16 frames are recorded
            image_perframe = read_tarfile(seg_path, tvad_idx)
            frames.append(image_perframe)
       
        # since the face bboxes and corresponding name ids are predicted for 16 frames, 
        # below takes the results from frame 0, 2, 4, ..., to form 8-frame results.
        bboxes = ast.literal_eval(bboxes)
        bboxes = bboxes[::2]
        name_ids = ast.literal_eval(name_ids)
        bboxes_filtered = []   # only bboxes corresponding to a recognised character is kept
        all_name_ids = {}  # in the form of {name_id_1: 1, name_id_2: 2, ...}
        for frame_idx in range(len(frames)):
            bboxes_filtered_per_frame = {}
            for name_idx, (name_id, bbox_idx_list) in enumerate(name_ids.items()):
                for bbox_idx in bbox_idx_list:
                    if bbox_idx[0] == int(frame_idx * 2) and name_id not in bboxes_filtered_per_frame.keys():
                        bboxes_filtered_per_frame[name_id] = bboxes[frame_idx][bbox_idx[1]]
                        if name_id not in all_name_ids.keys():
                            all_name_ids[name_id] = len(all_name_ids)
            bboxes_filtered.append(bboxes_filtered_per_frame)


        if self.label_type=="none":
            processed_frames = [expand2square(frame, tuple(int(x*255) for x in self.processor_mean)) for frame in frames]
        else:
            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: # skip as no bbox presents in this frame
                    processed_frames.append(expand2square(frame, tuple(int(x*255) for x in self.processor_mean)))
                    continue
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()

                        # overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))
                    processed_frame = expand2square(processed_frame, tuple(int(x*255) for x in self.processor_mean)) 
                    processed_frames.append(processed_frame)

        video_tensor = self.processor.preprocess(processed_frames, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, non_blocking=True)
        
        # Uncomment for visualisations
        # for frame_idx, processed_frame in enumerate(processed_frames):
        #     os.makedirs("tmp", exist_ok = True)
        #     filename = f"tmp/{imdbid}-{index}-{frame_idx}.jpg"
        #     processed_frame.save(filename)
        #     print(f"---Save file {filename} ---")

    
        # formulate the character name text prompt
        charbank_dict = self.charbank_dict[imdbid]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  #.split(" ")[0]
        
        if char_text == ". Possible characters (labeled by {label_type}): ": # no character recognised
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text="", label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)
        else:
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text= char_text.format(label_type=self.label_type), label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)

        return_dict =  {'video': video_tensor, \
                'imdbid': imdbid, \
                'input_id': input_id,
                'prompt': text_prompt,
                'gt_text': gt_text,
                'start': start, 
                'end': end, 
                'start_': start_, 
                'end_': end_, 
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['input_id'] = [sample['input_id'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        return out_batch



class CMDAD_FrameLoader():
    def __init__(self,
                tokenizer,
                processor,
                general_prompt,
                video_type, 
                label_type, 
                label_width, 
                label_alpha,
                anno_path,
                video_dir,
                charbank_path,
                **kwargs):
        self.processor = processor
        self.tokenizer = tokenizer
        self.general_prompt=general_prompt
        self.video_type = video_type
        self.processor_mean = self.processor.image_mean

        # label information, including colour coding, type, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha

        # load annotation file
        self.anno_df = pd.read_csv(anno_path)
        self.anno_df['num_words'] = self.anno_df.apply(lambda x: len(x['text'].strip().split()), axis=1)
        self.anno_df = self.anno_df[(self.anno_df['num_words'] < 64) & (self.anno_df['num_words'] > 1)]
        self.anno_df['num_string'] = self.anno_df.apply(lambda x: len(x['text']), axis=1)
        self.anno_df = self.anno_df[self.anno_df['num_string'] < 250]

        # prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        with open(os.path.join(charbank_path)) as fobj:
            charbank_dict = json.load(fobj)
        for key in charbank_dict.keys():
            self.charbank_dict[key] = {single_charbank_dict["id"]:single_charbank_dict["role"] for single_charbank_dict in charbank_dict[key]}    

        self.all_clips = []
        for anno_idx, anno_row in self.anno_df.iterrows():
            cmd_filename = anno_row['cmd_filename']
            video_path = os.path.join(video_dir, cmd_filename + '.mkv')
            if os.path.exists(video_path):
                self.all_clips.append((anno_row["imdbid"], video_path, anno_row["scaled_start"], anno_row["scaled_end"], anno_row["start"], anno_row["end"], anno_row["text"], anno_row["bboxes"], anno_row["pred_ids"]))
        print(f"In total {len(self.all_clips)} CMD-AD clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, video_path, start, end, start_, end_, gt_text, bboxes, name_ids = self.all_clips[index]
        
        # load 8 frames 
        frames = process_video(video_path, num_frames=8, start=start, end=end)
        
        # since the face bboxes and corresponding name ids are predicted for 16 frames, 
        # below takes the results from frame 0, 2, 4, ..., to form 8-frame results.
        bboxes = ast.literal_eval(bboxes)
        bboxes = bboxes[::2]
        name_ids = ast.literal_eval(name_ids)    
        bboxes_filtered = []   # only bboxes corresponding to a recognised character is kept
        all_name_ids = {}  # in the form of {name_id_1: 1, name_id_2: 2, ...}
        for frame_idx in range(len(frames)):
            bboxes_filtered_per_frame = {}
            for name_idx, (name_id, bbox_idx_list) in enumerate(name_ids.items()):
                for bbox_idx in bbox_idx_list:
                    if bbox_idx[0] == int(frame_idx * 2) and name_id not in bboxes_filtered_per_frame.keys():
                        bboxes_filtered_per_frame[name_id] = bboxes[frame_idx][bbox_idx[1]]
                        if name_id not in all_name_ids.keys():
                            all_name_ids[name_id] = len(all_name_ids)
            bboxes_filtered.append(bboxes_filtered_per_frame)


        if self.label_type=="none":
            processed_frames = [expand2square(frame, tuple(int(x*255) for x in self.processor_mean)) for frame in frames]
        else:
            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: # skip as no bbox presents in this frame
                    processed_frames.append(expand2square(frame, tuple(int(x*255) for x in self.processor_mean)))
                    continue
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()

                        # overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))
                    processed_frame = expand2square(processed_frame, tuple(int(x*255) for x in self.processor_mean)) 
                    processed_frames.append(processed_frame)

        video_tensor = self.processor.preprocess(processed_frames, return_tensors='pt')['pixel_values'].to(dtype=torch.float16, non_blocking=True)
        
        # Uncomment for visualisations
        # for frame_idx, processed_frame in enumerate(processed_frames):
        #     os.makedirs("tmp", exist_ok = True)
        #     filename = f"tmp/{imdbid}-{index}-{frame_idx}.jpg"
        #     processed_frame.save(filename)
        #     print(f"---Save file {filename} ---")

        # formulate the character name text prompt
        charbank_dict = self.charbank_dict[imdbid]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  #.split(" ")[0]
        
        if char_text == ". Possible characters (labeled by {label_type}): ": # no character recognised
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text="", label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)
        else:
            input_id, text_prompt = text_to_token(self.general_prompt.format(video_type=self.video_type, char_text=char_text.format(label_type=self.label_type), label_type=self.label_type, duration=str(round(end_-start_, 2))), self.tokenizer)


        return_dict =  {'video': video_tensor, \
                'imdbid': imdbid, \
                'input_id': input_id,
                'prompt': text_prompt,
                'gt_text': gt_text,
                'start': start, 
                'end': end, 
                'start_': start_, 
                'end_': end_, 
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['input_id'] = [sample['input_id'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        return out_batch
