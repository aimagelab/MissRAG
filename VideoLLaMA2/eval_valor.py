import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm    
import json
import re
import ast
from videollama2 import model_init, mm_infer_batch, mm_infer
import sys
from videollama2.dist_utils import get_rank
sys.path.append('./')

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
    if len(batch) == 0:
        return None, None, None
    video, audio, video_id = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)

    return video, audio, video_id

def parse_args():
    parser = argparse.ArgumentParser(description="Eval Valor")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "av"], help='', required=False, default='av'
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/VALOR-32K/VALOR-32K-annotations/valor-32k-annotations/desc_test.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/VALOR-32K/data'
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/eval_valor.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--prompt_template", type=str, default='Provide a detailed description for the given video in one sentence.'
    )
    args = parser.parse_args()
    return args

def setup_seeds(args):
    seed = args.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

class CaptionDataset(Dataset):
    def __init__(self, processor, data_path, root) -> None:
        super().__init__()
        self.datas = json.load(open(data_path))
        self.root = root
        self.processor = processor

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, _, _ = parse_video_id(data['video_id'])
        video_name = video_id + '.mp4'

        vpath = f"{self.root}/{video_id}/{video_name}"

        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            return None

        try:
            audio_video_tensor = self.processor(vpath, va=True) 
        except:
            print(f"Unable to open {vpath}")
            return None

        return audio_video_tensor['video'], audio_video_tensor['audio'], data['video_id']

 # ========================================

if __name__ == "__main__":
    # ========================================
    #             Model Initialization
    # ========================================

    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
        

    model, processor, tokenizer = model_init(args.model_path)
    model.eval()
    setup_seeds(args)
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    dataset = CaptionDataset(data_path=args.data_path, root=args.root, processor=processor['video'])   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    count = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, video_ids = data
            if video is None or audio is None:
                print("Data is None, skipping...")
                count += 1
                continue

            if args.modal_type == 'av':
                sample = {'video': video[0], 'audio': audio[0]}   
            elif args.modal_type == 'v':
                sample = video[0]
            elif args.modal_type == 'a':
                sample = audio[0]
            else:
                raise NotImplementedError
            
            prompt = args.prompt_template
             
            result = mm_infer(
                image_or_video=sample,
                instruct=prompt,
                model=model,
                tokenizer=tokenizer,
                modal='audio' if args.modal_type == "a" else "video",
                do_sample=False,
            )

            print(f"Answer: {result}")
            predictions.append({'image_id': video_ids[0], 'caption': result})
            
            # Save every 1000 predictions
            count += 1
            if count % 1000 == 0:
                print(f"Saving predictions at {count}")
                answer_path = args.answer_path.replace('.json', f'_{count}.json')
                with open(answer_path, 'w') as f:
                    json.dump(predictions, f)

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)