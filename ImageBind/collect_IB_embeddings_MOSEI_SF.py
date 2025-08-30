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
from util.misc import setup_for_distributed
import numpy as np
import argparse
from util.misc import get_random_free_port
import pandas as pd
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

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
        
       
        text = imagebind_data.load_and_transform_text([text], "cpu")[0]
        try:
            video = imagebind_data.load_and_transform_video_data([video_path], "cpu")[0]
        except:
            print(f"Unable to open {video_path}")
            return None
        try:
            audio = imagebind_data.load_and_transform_audio_data([audio_path], "cpu")[0]
        except:
            print(f"Unable to open {audio_path}")
            return None

        return video, audio, text, id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSEI"
    )
    parser.add_argument(
        "--mode", type=str, default="train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--answer_path", type=str, default='/path/to/MissRAG/ImageBind/IB_embeddings/train/mosei_SF'
    )
    parser.add_argument("--debug", action='store_true', help="debug, don't use model but fake data")
    args = parser.parse_args()  
    
    # os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
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

    # print("Loading ImageBind...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("ImageBind loaded!")

    dataset = MOSEI(root=args.root, mode=args.mode)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False,
                            pin_memory=True,
                            num_workers=8)
    num_samples = len(dataset)
    if args.debug:
        num_samples = 6
    batch_size = args.batch_size
    index = 0
    
    for video, audio, text, id in tqdm(dataloader):
        # Move data to the appropriate device if using GPUs
        # Load data
        inputs = {
            ModalityType.VISION: video.to(device),
            ModalityType.AUDIO: audio.to(device),
            ModalityType.TEXT: text.to(device)
        }

        
        with torch.no_grad():
            embeddings = model(inputs)

        video_tokens_np = embeddings[ModalityType.VISION].cpu().numpy()
        audio_tokens_np = embeddings[ModalityType.AUDIO].cpu().numpy()
        text_tokens_np = embeddings[ModalityType.TEXT].cpu().numpy()
        batch_size_actual = video_tokens_np.shape[0] 
        torch.save(video_tokens_np, f"{args.answer_path}/mosei_video_IB_embeddings_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/mosei_audio_IB_embeddings_{index}.pt")
        torch.save(text_tokens_np,  f"{args.answer_path}/mosei_text_IB_embeddings_{index}.pt")
        torch.save(id,              f"{args.answer_path}/mosei_ids_{index}.pt")   
        # Update the index for the next batch
        index += batch_size_actual  
  
        if args.debug:
            break

    print("DONE")


 
    

