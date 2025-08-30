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
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="MOSI Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSI"
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=[]
    )
    parser.add_argument(
        "--use_text_modality", action='store_true', default=False, help='Enable or disable the text input modality. Accepts true/false.'
    )
    parser.add_argument(
        "--missing_prompt", action='store_true', default=False
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results_chatbridge/eval_music_avqa.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=4
    )
    parser.add_argument(
        "--prompt_template", type=str, default='Given the class set ["Positive", "Negative"] What is the sentiment of this video?'
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

class MOSI(Dataset):
    def __init__(self, vis_processor, aud_processor, text_processor=None, mode: str = "test", root: str = "/path/to/MOSI") -> None:
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
    #text_processor = BlipQuestionProcessor()

    dataset = MOSI(root=args.root, vis_processor=vis_processor, aud_processor=aud_processor)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    print('Initialization Finished')

    with open(args.prompt_file_path, 'r') as file:
        prompt_file = json.load(file)

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

            video = video.permute(0,2,1,3,4)
            samples = {}

            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()

            prompts = []
            if args.missing_prompt:
                miss_prompt = ' ' + get_miss_prompt(modal=args.modal, task_modals=['video', 'audio', 'text'], use_text_modality=args.use_text_modality)
            else:
                miss_prompt=''

            for text in texts:
                conv = CONV_VIDEO.copy()
                prompt_template = random.choice(prompt_file["tva-sa"])
                
                if 'video' in args.modal and 'audio' in args.modal and args.use_text_modality: #video+audio+text
                    prefix = "Given following video: <query>, its background audio: <query> and the input text: <TEXT>.".replace('<TEXT>', text)
                    if len(prompts)==0:
                        samples['task'] = 'tva'

                if 'audio' in args.modal and 'video' in args.modal and not args.use_text_modality: #audio+video
                    prefix = "Given following video: <query> and its background audio: <query>"
                    if len(prompts)==0:
                        samples['task'] = 'tva'

                if 'audio' in args.modal and 'video' not in args.modal and args.use_text_modality: #audio+text
                    prefix = "Given following audio: <query> and input text: <TEXT>.".replace('<TEXT>', text)
                    if len(prompts)==0:
                        samples['task'] = 'ta'

                if 'video' in args.modal and 'audio' not in args.modal and args.use_text_modality: #video+text
                    prefix = "Given following video: <query> and input text: <TEXT>.".replace('<TEXT>', text)
                    if len(prompts)==0:
                        samples['task'] = 'tv'

                if 'audio' in args.modal and 'video' not in args.modal and not args.use_text_modality: #audio
                    prefix = "Given following audio: <query>."
                    if len(prompts)==0:
                        samples['task'] = 'ta'

                if 'video' in args.modal and 'audio' not in args.modal and not args.use_text_modality: #video
                    prefix = "Given following video: <query>."
                    if len(prompts)==0:
                        samples['task'] = 'tv'

                if 'audio' not in args.modal and 'video' not in args.modal and args.use_text_modality: #text
                    prefix = "Given following input text: <TEXT>.".replace('<TEXT>', text)
                    if len(prompts)==0:
                        samples['task'] = 'only-t'

                conv.append_message(conv.roles[0], prefix+miss_prompt)  
                conv.append_message(conv.roles[0], prompt_template)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())
            
            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)
            
            samples['conversation'] = prompts
             
            results = model.forward_inference(samples)
        
            for video_id, result in zip(id, results): 
                predictions.append({
                    'image_id': video_id,
                    'classification': result.strip(),    
                })
                
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)