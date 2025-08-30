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
import pandas as pd
sys.path.append('./')


def parse_args():
    parser = argparse.ArgumentParser(description="MOSI Eval")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "t", "av", "at", "vt", "avt"], help='', required=False, default='avt'
    )
    parser.add_argument(
        "--user_command",
        type=str,
        default="""Given the class set ["Positive", "Neutral", "Negative"], what is the sentiment of this video?.""",
        help='Template for the prompt'
    )
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSI"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/eval_mosei.json"
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
    parser.add_argument("--debug", action='store_true', help="debug mode on")
    parser.add_argument("--prototype_prompt", action='store_true', default=False)
    args = parser.parse_args()
    return args

def setup_seeds(args):
    seed = args.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

class MOSI(Dataset):
    def __init__(self, processor, mode: str = "test", root: str = "/path/to/MOSI") -> None:
        super().__init__()
        self.mode = mode
        self.data = pd.read_csv(root+'/labels.csv')
        self.data['id'] = self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)
        self.data = self.data[self.data['mode'] == self.mode]
        self.root = root
        self.video_prefix = 'Raw'
        self.audio_prefix = 'Raw_audio'
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.data.iloc[index]['id']
        video_id = str(self.data.iloc[index]['video_id'])
        clip_id = str(self.data.iloc[index]['clip_id'])
        text, label, annotation = self.data.iloc[index]['text'], self.data.iloc[index]['label'], self.data.iloc[index]['annotation']

        vpath = os.path.join(self.root, self.video_prefix, video_id, clip_id + '.mp4')
        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception
        
        audio_video_tensor = self.processor(vpath, va=True)

        sample = {
            'id': id,
            'video': audio_video_tensor['video'],
            'audio': audio_video_tensor['audio'],
            'video_id': video_id,
            'clip_id': clip_id,
            'text': text,
            'label': label,
            'annotation': annotation
        }

        return sample

 # ========================================

if __name__ == "__main__":
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
    dataset = MOSI(processor=processor['video'], root=args.root)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            id = data['id']
            video = data['video']
            audio = data['audio']
            video_id = data['video_id']
            clip_id = data['clip_id']
            texts = data['text']
            label = data['label']
            annotation = data['annotation']
            print(f"id:{id}")

            if args.modal_type == 'avt' or args.modal_type == 'av':
                sample = {'video': video[0], 'audio': audio[0]} 
                modal = "video"  
            
            elif args.modal_type == 'v' or args.modal_type == 'vt':
                sample = video[0]
                modal = "video"
            
            elif args.modal_type == 'a' or args.modal_type == 'at':
                sample = audio[0]
                modal = "audio"
            
            elif args.modal_type == 't':
                sample = None
                modal = "text"
            else:
                raise NotImplementedError
            
            for text in texts:
                if "t" in args.modal_type:
                    text_i_p = """Input text: {text}. """.format(text=text)
                    question = text_i_p + args.user_command         
                else:
                    question = args.user_command
            
            result = mm_infer(
                image_or_video=sample,
                instruct=question,
                model=model,
                tokenizer=tokenizer,
                modal=modal,
                do_sample=False,
            )

            print(f"Answer: {result}")

            predictions.append({
                'image_id': id[0],
                'classification': result.strip(),  
            })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4) 