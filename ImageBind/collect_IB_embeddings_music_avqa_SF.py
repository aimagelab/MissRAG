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
from typing import List
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class AVQADataset(Dataset):
    def __init__(self, data_path, root) -> None:
        super().__init__()
        self.root = root 
        self.datas = json.load(open(data_path))
        self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = str(data['video_id'])
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        image_path = os.path.join(self.root, video_name)
        audio_path = os.path.join(self.root, audio_name)

        if not os.path.isfile(audio_path):
            print("File {} does not exist".format(audio_path))
            raise Exception
        
        if not os.path.isfile(image_path):
            print("File {} does not exist".format(image_path))
            raise Exception
        
        question_id = data['question_id']
        video_id = data['video_id']
        inputs = {
            ModalityType.VISION: imagebind_data.load_and_transform_video_data([image_path], "cpu")[0],
            ModalityType.AUDIO: imagebind_data.load_and_transform_audio_data([audio_path], "cpu")[0],
        }
        return inputs, question_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default='/path/to/MUSIC_AVQA_git/MUSIC-AVQA/data/json/avqa-train.json'
    )
    parser.add_argument(
        "--answer_path", type=str, default='/path/to/MissRAG/ImageBind/IB_embeddings/train/music_avqa_SF'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/MUSIC-AVQA'
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
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

    dataset = AVQADataset(data_path=args.data_path, root=args.root)
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
    
    for inputs, question_id in tqdm(dataloader):
        # Move data to the appropriate device if using GPUs
        # Load data
        inputs = {
            ModalityType.VISION: inputs[ModalityType.VISION].to(device),
            ModalityType.AUDIO: inputs[ModalityType.AUDIO].to(device),
        }

        
        with torch.no_grad():
            embeddings = model(inputs)

        video_tokens_np = embeddings[ModalityType.VISION].cpu().numpy()
        audio_tokens_np = embeddings[ModalityType.AUDIO].cpu().numpy()
        batch_size_actual = video_tokens_np.shape[0] 
        torch.save(video_tokens_np, f"{args.answer_path}/music_avqa_video_IB_embeddings_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/music_avqa_audio_IB_embeddings_{index}.pt")
        torch.save(question_id, f"{args.answer_path}/music_avqa_question_ids_{index}.pt")   
        # Update the index for the next batch
        index += batch_size_actual  
  
        if args.debug:
            break
    print("DONE")


 
    

