
import sys
sys.path.append('./')
import argparse
import os
import random
import numpy as np
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
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
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
        "--answer_path", type=str, default="/path/to/MissRAG/ChatBridge/prototypes/data/modality_tokens/train/charadesego_SF"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    #parser.add_argument(
    #    "--missing_prompt", type=bool, default=False
    #)
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


def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Filter out None entries
    video, audio, video_id = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)

    return video, audio, video_id

class CaptionDataset(Dataset):
    def __init__(self, vis_processor, aud_processor, data_path, video_path, audio_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path
        self.datas=[]
        with open(data_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.datas.append(row['id'])
        #self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}
        self.vis_processor = vis_processor
        self.aud_processor = aud_processor

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
            return None  # Indicate that this item should be skipped
        
        if not os.path.isfile(audio_path):
            print(f"Warning: File {audio_path} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        try:
            frms = self.vis_processor(video_path)
            auds = self.aud_processor(audio_path)
        except:
            print(f"Unable to open {video_path}")
            return None

        return frms, auds, video_id
 # ========================================

if __name__ == "__main__":
    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
        
    random.seed(args.seed)

    #cfg = Config(args)
    config = OmegaConf.load(args.cfg_path)
    setup_seeds(config)
    #model_config.device_8bit = args.gpu_id
    # model_cls = registry.get_model_class(config.model.arch)
    model = ChatBridge_OME.from_config(config.model)
    model.cuda()
    model.eval()
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=4)
    aud_processor = BlipAudioEvalProcessor()
    dataset = CaptionDataset(vis_processor=vis_processor, aud_processor=aud_processor, data_path=args.data_path, video_path=args.video_path, audio_path=args.audio_path)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    correct = 0
    index = 0
    with torch.no_grad():
        for video, audio, video_ids in tqdm(dataloader):
            
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
                print(video_ids[i], image_tok_np[i].shape, audio_tok_np[i].shape)

            torch.save(image_tok_np, f"{args.answer_path}/charadesego_video_tokens_{index}.pt")
            torch.save(audio_tok_np, f"{args.answer_path}/charadesego_audio_tokens_{index}.pt")
            torch.save(video_ids, f"{args.answer_path}/charadesego_video_ids_{index}.pt")

            index += batch_size_actual

    print("DONE")