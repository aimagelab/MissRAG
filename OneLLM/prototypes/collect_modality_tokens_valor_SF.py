import sys
sys.path.append('./')
import os
import json
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from fairscale.nn.model_parallel import initialize as fs_init
from util.misc import default_tensor_type
from util.misc import setup_for_distributed
import numpy as np
from model.meta import MetaModel
from data.conversation_lib import conv_templates
from data import video_utils
from data.data_utils import make_audio_features
import argparse
from util.misc import get_random_free_port
from typing import List
import h5py

def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

def parse_video_id(video_data):
    """
    Parse video_id and extract video_id, start_time, and end_time.
    
    Args:
        video_data (str): The full video_id string in the format "video_id_startTime_endTime".
        
    Returns:
        tuple: (video_id, start_time, end_time)
        
    Raises:
        ValueError: If the video_data does not contain at least three parts separated by underscores.
    """
    parts = video_data.split('_')
    
    if len(parts) < 3:
        raise ValueError("Invalid video_data format: Expected at least 3 parts separated by underscores.")
    
    start_time, end_time = parts[-2], parts[-1]
    video_id = '_'.join(parts[:-2])
    
    return video_id, start_time, end_time


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None entries
    video, audio, video_id = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)

    return video, audio, video_id


class CaptionDataset(Dataset):
    def __init__(self, data_path, root) -> None:
        super().__init__()
        super().__init__()
        self.datas = json.load(open(data_path))
        self.root = root
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, _, _ = parse_video_id(data['video_id'])
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        image_path = f"{self.root}/{video_id}/{video_name}"
        audio_path = f"{self.root}/{video_id}/{audio_name}"

        if not os.path.isfile(image_path):
            print(f"Warning: File {image_path} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        
        if not os.path.isfile(audio_path):
            print(f"Warning: File {audio_path} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        try:
            image = load_video(image_path)
            audio = load_audio(audio_path)
        except:
            print(f"Unable to open {image_path}")
            return None

        return image, audio, data['video_id']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/path/to/MissRAG/OneLLM/pretrained/consolidated.00-of-01.pth"
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/VALOR-32K/VALOR-32K-annotations/valor-32k-annotations/desc_train.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/VALOR-32K/data_train'
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens/train/valor_SF"
    )
    parser.add_argument("--debug", action='store_true', help="debug, don't use model but fake data")
    args = parser.parse_args()  

    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 

    mp.set_start_method("spawn")
    port = get_random_free_port()
    dist.init_process_group(
        backend="nccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:{port}")
    fs_init.initialize_model_parallel(1)
    torch.cuda.set_device(0)
    torch.manual_seed(1)
    np.random.seed(1)
    # set the print behavior.
    setup_for_distributed(True)

    target_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16
    }['fp16']


    with default_tensor_type(dtype=target_dtype, device="cuda"):
        model = MetaModel("onellm", "config/llama2/7B.json", None, "config/llama2/tokenizer.model")
    if args.debug is False:
        print("Loading pretrained weights ...")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
        model.half().cuda()
        model.eval()
        print(f"Model = {str(model)}")

    dataset = CaptionDataset(data_path=args.data_path, root=args.root)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=custom_collate_fn)
    
    num_samples = len(dataset)
    if args.debug:
        num_samples = 6
    batch_size = args.batch_size
    index = 0
    
    for video, audio, video_ids in tqdm(dataloader):
        print(video.shape, audio.shape, video_ids)
        
        # Move data to the appropriate device if using GPUs
        video = video.cuda().to(target_dtype)
        audio = audio.cuda().to(target_dtype)
        
        # Compute modality tokens
        if args.debug:
            video_modality_tokens = torch.randn(video.shape[0], 30, 4096)
            audio_modality_tokens = torch.randn(audio.shape[0], 30, 4096)
        else:
            with torch.inference_mode():
                video_modality_tokens = model.llma.encode_image(video, modal="video")
                audio_modality_tokens = model.llma.encode_image(audio, modal="audio")
        print(video_modality_tokens.shape, audio_modality_tokens.shape, video_ids)
        
        # Move tensors to CPU and convert to NumPy arrays
        video_tokens_np = video_modality_tokens.cpu().numpy()
        audio_tokens_np = audio_modality_tokens.cpu().numpy()
        
        batch_size_actual = video_tokens_np.shape[0]     
        
        for i in range(batch_size_actual):
            print(video_ids[i], video_tokens_np[i].shape, audio_tokens_np[i].shape)
            
        torch.save(video_tokens_np, f"{args.answer_path}/valor_video_tokens_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/valor_audio_tokens_{index}.pt")
        torch.save(video_ids, f"{args.answer_path}/valor_video_ids_{index}.pt")   

        index += batch_size_actual  
  
        if args.debug:
            break
        
    print("DONE")


 
    

