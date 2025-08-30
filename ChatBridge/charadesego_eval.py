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
from chatbridge.conversation.conversation_lib import get_miss_prompt
from chatbridge.processors.blip_processors import BlipAudioEvalProcessor, BlipQuestionProcessor
from chatbridge.processors.alpro_processors import AlproVideoEvalProcessor
from chatbridge.common.registry import registry
from tqdm import tqdm    
import json
from omegaconf import OmegaConf
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="CharadesEGO Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--data_path", type=str, default='/path/to/Charades/CharadesEgo/CharadesEgo_v1_test_only3rd.csv'
    )
    parser.add_argument(
        "--video_path", type=str, default='/path/to/Charades/CharadesEgo_v1'
    )
    parser.add_argument(
        "--audio_path", type=str, default='/path/to/Charades/CharadesEgo_v1_Audio_Extracted'
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=[]
    )
    parser.add_argument(
        "--missing_prompt", action='store_true', default=False
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results_chatbridge/eval_charadesego_cap.json"
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
        "--index_prompt", type=int, default=10
    )
    parser.add_argument(
        "--prompt_template", type=str, default='Based on the video and audio, could you provide a short answer to the question:'
    )
    parser.add_argument(
        "--prompt_file_path", type=str, default='instructiontuning_configs/task_prompt_eval.json'
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
    def __init__(self, test_path, video_path, audio_path, vis_processor, aud_processor) -> None:
        super().__init__()
        self.vis_processor = vis_processor
        self.aud_processor = aud_processor

        self.video_path = video_path
        self.audio_path = audio_path    
        self.datas=[]

        with open(test_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.datas.append(row['id'])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        video_id = self.datas[index]
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        vpath = f"{self.video_path}/{video_name}"
        apath = f"{self.audio_path}/{audio_name}"

        if not os.path.isfile(vpath):
            print(f"Warning: File {vpath} is missing")
            raise Exception
        
        if not os.path.isfile(apath):
            print(f"Warning: File {apath} is missing")
            raise Exception
        try:
            frms = self.vis_processor(vpath)
            auds = self.aud_processor(apath)
        except:
            print(f"Unable to open {vpath}")
            raise Exception

        return frms, auds, video_id
 # ========================================

if __name__ == "__main__":
    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
        

    #cfg = Config(args)
    config = OmegaConf.load(args.cfg_path)
    setup_seeds(args)
    #model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(config.model.arch)
    model = model_cls.from_config(config.model)
    model.cuda()
    model.eval()
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=4)
    aud_processor = BlipAudioEvalProcessor()
    dataset = CaptionDataset(test_path=args.data_path, video_path=args.video_path, audio_path=args.audio_path, vis_processor=vis_processor, aud_processor=aud_processor)      
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print('Initialization Finished')
    
    with open(args.prompt_file_path, 'r') as file:
        prompt_file = json.load(file)

    print("Starting...")
    outputs = []
    with torch.no_grad():
        for video, audio, video_ids in tqdm(dataloader):
            video = video.permute(0,2,1,3,4)
            samples = {}
            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()
            
            prompts = []

            if args.missing_prompt:
                miss_prompt = ' ' + get_miss_prompt(modal=args.modal, task_modals=['video', 'audio'])
            else:
                miss_prompt=''

            bsz = video.shape[0]

            for question in range(bsz):
                conv = CONV_VIDEO.copy()
                prompt_template = prompt_file["tva-cap"][args.index_prompt]

                if 'video' in args.modal and 'audio' in args.modal:
                    prefix = "Given following video: <query> and its background audio: <query>."
                    if len(prompts)==0:
                        samples['task'] = 'tva'

                if 'video' in args.modal and 'audio' not in args.modal:
                    prefix = "Given following video: <query>."
                    if len(prompts)==0:
                        samples['task'] = 'tv'

                if 'audio' in args.modal and 'video' not in args.modal:
                    prefix = "Given following audio: <query>."
                    if len(prompts)==0:
                        samples['task'] = 'ta'

                conv.append_message(conv.roles[0], prefix+miss_prompt)  
                conv.append_message(conv.roles[0], prompt_template)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())
            
            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)
            
            samples['conversation'] = prompts
             
            results = model.forward_inference(samples)
        
            for video_id, result in zip(video_ids, results):
                outputs.append({
                    'image_id': video_id,
                    'caption': result.strip()
                })
    with open(args.answer_path, 'w') as f:
        json.dump(outputs, f)