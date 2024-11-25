import os
import copy
import torch
import tarfile
import numpy as np

from PIL import Image
from io import BytesIO


def read_tarfile(tar_path, img_idx):
    """
    Input: tar_file and image_index (start from 1)
    Output: PIL image
    """
    dirname = os.path.basename(tar_path.replace(".tar", ""))
    imagename = os.path.join(dirname, str(img_idx).zfill(5) + ".jpg")
    with tarfile.open(tar_path, 'r') as tar:
        fileobj = tar.extractfile(imagename)
        image_data = fileobj.read()
        image_tmp = Image.open(BytesIO(image_data))
        image = copy.deepcopy(image_tmp)
        image_tmp.close()
        fileobj.close()
    return image

def process_video(decord_vr, num_frames=16, start = None, end = None):
    """
    Input: decord video reader file
    Output: list of (16) numpy frames
    """
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    else:
        print("check start and end")
        sys.exit()
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()    
    images = [f.numpy() if isinstance(f, torch.Tensor) else f for f in video_data]
    return images

def group_and_filter_id(pred_id_all_frames, pred_cos_all_frames, score_threshold):
    """
    Input: predicted id for all frames: [["nmxxxxxxx", ...], ...]; predicted cosine similarity scores for all frames: [[0.3, ...], ...]; threshold for filtering
    Output: dictionary of name ids: {}"nmxxxxxxx": [(0, 0), ...], ...}
    """
    index_filtered_dict = {}
    for frame_idx, pred_cos_per_frame in enumerate(pred_cos_all_frames):
        pred_id_per_frame = pred_id_all_frames[frame_idx]
        recorded_id_dict = {}
        for face_idx, pred_cos in enumerate(pred_cos_per_frame):
            if pred_cos < score_threshold: # ignore faces with low cosine similarity (unconfident recognition)
                continue
            pred_id = pred_id_per_frame[face_idx]
            if pred_id not in recorded_id_dict.keys():
                recorded_id_dict[pred_id] = [pred_cos, (frame_idx, face_idx)]
            else:
                if pred_cos > recorded_id_dict[pred_id][0]: # match the face with the character based on highest cosine similarity
                    recorded_id_dict[pred_id] = [pred_cos, (frame_idx, face_idx)]
        # collect up information in this frame and add it to the final result
        for result_id, result_value in recorded_id_dict.items():
            if result_id not in index_filtered_dict.keys():
                index_filtered_dict[result_id] = [result_value[1]]
            else:
                index_filtered_dict[result_id].append(result_value[1])
    return index_filtered_dict

