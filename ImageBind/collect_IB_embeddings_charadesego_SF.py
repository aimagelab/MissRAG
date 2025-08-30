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
import argparse
from util.misc import get_random_free_port
import csv
from imagebind import data as imagebind_data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None entries
    video, audio, video_id = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)
    return video, audio, video_id
    
class CaptionDataset(Dataset):
    def __init__(self, data_path, video_path, audio_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path  
        self.datas=[]
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.datas.append(row['id'])
                    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        video_id = self.datas[index]

        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'

        video_path = f"{self.video_path}/{video_name}"
        audio_path = f"{self.audio_path}/{audio_name}"

        if not os.path.isfile(video_path):
            print(f"Warning: File {video_path} is missing. Skipping...")
            raise Exception
        
        if not os.path.isfile(audio_path):
            print(f"Warning: File {audio_path} is missing. Skipping...")
            raise Exception
         
        try:
            video = imagebind_data.load_and_transform_video_data([video_path], "cpu")[0]
            audio = imagebind_data.load_and_transform_audio_data([audio_path], "cpu")[0]
        except:
            print(f"Unable to open {video_path}")
            return None        

        return video, audio, video_id    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, default='/path/to/Charades/CharadesEgo/CharadesEgo_v1_train_only3rd.csv'
    )
    parser.add_argument(
        "--video_path", type=str, default='/path/to/Charades/CharadesEgo_v1'
    )
    parser.add_argument(
        "--audio_path", type=str, default='/path/to/Charades/CharadesEgo_v1_Audio_Extracted'
    )
    parser.add_argument(
        "--answer_path", type=str, default='/path/to/MissRAG/ImageBind/IB_embeddings/train/charadesego_SF'
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

    print("Loading ImageBind...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    print("ImageBind loaded!")

    dataset = CaptionDataset(data_path=args.data_path, video_path=args.video_path, audio_path=args.audio_path)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False,
                            pin_memory=True,
                            num_workers=8, 
                            collate_fn=custom_collate_fn)
    num_samples = len(dataset)
    if args.debug:
        num_samples = 6
    batch_size = args.batch_size
    index = 0

    for video, audio, video_id in tqdm(dataloader):
        # Move data to the appropriate device if using GPUs
        # Load data
        inputs = {
            ModalityType.VISION: video.to(device),
            ModalityType.AUDIO: audio.to(device),
        }

        
        with torch.no_grad():
            embeddings = model(inputs)

        video_tokens_np = embeddings[ModalityType.VISION].cpu().numpy()
        audio_tokens_np = embeddings[ModalityType.AUDIO].cpu().numpy()
        batch_size_actual = video_tokens_np.shape[0] 
        torch.save(video_tokens_np, f"{args.answer_path}/charadesego_video_IB_embeddings_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/charadesego_audio_IB_embeddings_{index}.pt")
        torch.save(video_id, f"{args.answer_path}/charadesego_video_ids_{index}.pt")   
        # Update the index for the next batch
        index += batch_size_actual  
  
        if args.debug:
            break
    print("DONE")


 
    

