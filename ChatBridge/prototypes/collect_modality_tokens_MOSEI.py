
import sys
sys.path.append('./')
import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from chatbridge.common.config import Config
from chatbridge.common.dist_utils import get_rank
from chatbridge.conversation.conversation import CONV_VIDEO
from chatbridge.processors.blip_processors import BlipAudioEvalProcessor, BlipQuestionProcessor
from chatbridge.processors.alpro_processors import AlproVideoEvalProcessor
from chatbridge.models.chatbridge_only_modality_encoders import ChatBridge_OME
from chatbridge.common.registry import registry
from tqdm import tqdm    
import json
from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSEI"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/ChatBridge/prototypes/data/modality_tokens/train/MOSEI_SF"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--prompt_template", type=str, default='Based on the video and audio, could you provide a short answer to the question:'
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


class MOSEI(Dataset):
    def __init__(self, vis_processor, aud_processor, text_processor=None, mode: str = "train", root: str = "/path/to/MOSEI") -> None:
        super().__init__()
        self.mode = mode
        self.data = pd.read_csv(root+'/labels.csv')
        self.data['id'] = self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)
        self.data = self.data[self.data['mode'] == self.mode]
        self.root = root
        self.video_prefix = 'Raw'
        self.audio_prefix = 'Raw_audio'
        self.vis_processor = vis_processor
        self.aud_processor = aud_processor
        self.text_processor = text_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        id = self.data.iloc[index]['id']
        video_id = str(self.data.iloc[index]['video_id'])
        clip_id = str(self.data.iloc[index]['clip_id'])
        text, label, annotation = self.data.iloc[index]['text'], self.data.iloc[index]['label'], self.data.iloc[index]['annotation']

        vpath = os.path.join(self.root, self.video_prefix, video_id, clip_id + '.mp4')
        apath = os.path.join(self.root, self.audio_prefix, video_id, clip_id + '.wav')

        if not os.path.isfile(apath):
            print("File {} does not exist".format(apath))
            raise Exception
        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception
        try:
            frms = self.vis_processor(vpath)
            auds = self.aud_processor(apath)
        except Exception as e:
            print(f"Error loading {vpath} or {apath}")
            print(e)
            return self.__getitem__(index+1)
        
        sample = {
            'id': id,
            'video': frms,
            'audio': auds,
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
        
    random.seed(args.seed)
    config = OmegaConf.load(args.cfg_path)
    setup_seeds(config)
    model = ChatBridge_OME.from_config(config.model)
    model.cuda()
    model.eval()
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=4)
    aud_processor = BlipAudioEvalProcessor()
    dataset = MOSEI(vis_processor=vis_processor, aud_processor=aud_processor, root=args.root)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    print('Initialization Finished')


    print("Starting...")
    predictions = []
    correct = 0
    index = 0
    with torch.no_grad():
        for sample in tqdm(dataloader):
            id = sample['id']
            video = sample['video']
            audio = sample['audio']
            video_id = sample['video_id']
            clip_id = sample['clip_id']
            text = sample['text']
            label = sample['label']
            annotation = sample['annotation']
            
            video = video.permute(0,2,1,3,4)
            samples = {}
            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()

            image_tok = model.encode_img(samples["image"])[0]
            audio_tok = model.encode_audio(samples["audio"])[0]        

            image_tok_np = image_tok.cpu().numpy()
            audio_tok_np = audio_tok.cpu().numpy()

            batch_size_actual = image_tok_np.shape[0]

            for i in range(batch_size_actual):
                print(id[i], image_tok_np[i].shape, audio_tok_np[i].shape)

            torch.save(image_tok_np, f"{args.answer_path}/MOSEI_video_tokens_{index}.pt")
            torch.save(audio_tok_np, f"{args.answer_path}/MOSEI_audio_tokens_{index}.pt")
            torch.save(text, f"{args.answer_path}/MOSEI_text_{index}.pt")
            torch.save(id, f"{args.answer_path}/MOSEI_ids_{index}.pt")   

            index += batch_size_actual

    print("DONE")