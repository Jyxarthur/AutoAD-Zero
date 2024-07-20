import os
import cv2
import sys
import copy
import torch
import tarfile
import numpy as np

from PIL import Image
from io import BytesIO
from decord import VideoReader, cpu

# modify the path below
sys.path.append("/path/to/VideoLLaMA2/")
from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token


def process_video(video_path, num_frames=8, sample_scheme='uniform', start = None, end = None):
    """
    Input: video path, start & end time
    Output: a list of 8 consecutive PIL images
    """
    decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, num_frames, mode=sample_scheme, local_fps=local_fps)
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
    return images

def frame_sample(duration, num_frames, mode='uniform', local_fps=None):
    if mode == 'uniform':
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert local_fps is not None
        segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def expand2square(pil_img, background_color):
    """
    Pad the PIL image with indicated colour
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def convert_bounding_box_to_rectangle(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary bbox mask
    """
    x0, y0, x1, y1 = bbox

    # extend the bbox by 20%
    delta_x = (x1 - x0) * 0.1
    delta_y = (y1 - y0) * 0.1

    x0 = int(max(x0 - delta_x, 0))
    y0 = int(max(y0 - delta_y, 0))
    x1 = int(min(x1 + delta_x, canvas_width))
    y1 = int(min(y1 + delta_y, canvas_height))
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas[y0:y0+line_width, x0:x1] = 1. 
    canvas[y1-line_width:y1, x0:x1] = 1. 
    canvas[y0:y1, x0:x0+line_width] = 1.  
    canvas[y0:y1, x1-line_width:x1] = 1. 
    return canvas


def convert_bounding_box_to_ellipse(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary ellipse mask
    """
    x0, y0, x1, y1 = bbox
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # extend the ellipse by 25%
    axis_x = int(abs(x1 - x0) // 2 * 1.25)
    axis_y = int(abs(y1 - y0) // 2 * 1.25)
    
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas = cv2.ellipse(canvas, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1, line_width)
    return canvas

def text_to_token(question, tokenizer):
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    conv_mode = 'llama_2'
    question = default_mm_token + "\n" + question
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_MMODAL_token(prompt, tokenizer, modal_token_index, return_tensors='pt')
    return input_ids, prompt


def read_tarfile(tar_path, img_idx):
    """
    Input: tarfile path, image index to extract
    Output: an PIL image
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



