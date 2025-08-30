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
# from data.conversation_lib import conv_templates
from data import video_utils
from data.data_utils import make_audio_features
import argparse
from util.misc import get_random_free_port
from typing import List
import h5py
import csv
import pandas as pd


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

class MOSEI(Dataset):
    def __init__(self, 
                 mode: str = "train",
                 root: str = "/path/to/MOSEI") -> None:
        super().__init__()
        self.mode = mode
        self.data = pd.read_csv(root+'/labels.csv')
        self.data['id'] = self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)
        self.data = self.data[self.data['mode'] == self.mode]
        self.root = root
        self.video_prefix = 'Raw'
        self.audio_prefix = 'Raw_audio'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.data.iloc[index]['id']
        print(f"get_item: {id}")
        video_id = str(self.data.iloc[index]['video_id'])
        clip_id = str(self.data.iloc[index]['clip_id'])
        text, label, annotation = self.data.iloc[index]['text'], self.data.iloc[index]['label'], self.data.iloc[index]['annotation']

        video_path = os.path.join(self.root, self.video_prefix, video_id, clip_id + '.mp4')
        audio_path = os.path.join(self.root, self.audio_prefix, video_id, clip_id + '.wav')

        if not os.path.isfile(video_path):
            print("File {} does not exist".format(video_path))
            raise Exception
        
        if not os.path.isfile(audio_path):
            print("File {} does not exist".format(audio_path))
            raise Exception
        
        try:
            video = load_video(video_path)
            audio = load_audio(audio_path)
        except Exception as e:
            print(f"Error loading {video_path} or {audio_path}")
            print(e)
            return None

        return video, audio, text, id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/path/to/MissRAG/OneLLM/pretrained/consolidated.00-of-01.pth"
    )
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSEI"
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens/train/MOSEI_SF"
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

    dataset = MOSEI(root=args.root)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    num_samples = len(dataset)
    if args.debug:
        num_samples = 6
    batch_size = args.batch_size
    index = 0
    
    for video, audio, text, id in tqdm(dataloader):
        print(video.shape, audio.shape, id)
        
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
        print(video_modality_tokens.shape, audio_modality_tokens.shape, id)
        
        # Move tensors to CPU and convert to NumPy arrays
        video_tokens_np = video_modality_tokens.cpu().numpy()
        audio_tokens_np = audio_modality_tokens.cpu().numpy()
        
        batch_size_actual = video_tokens_np.shape[0]     
        
        for i in range(batch_size_actual):
            print(id[i], video_tokens_np[i].shape, audio_tokens_np[i].shape)
            
        torch.save(video_tokens_np, f"{args.answer_path}/MOSEI_video_tokens_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/MOSEI_audio_tokens_{index}.pt")
        torch.save(text, f"{args.answer_path}/MOSEI_text_{index}.pt")
        torch.save(id, f"{args.answer_path}/MOSEI_ids_{index}.pt")   

        index += batch_size_actual  
  
        if args.debug:
            break

    print("DONE")


 
    

