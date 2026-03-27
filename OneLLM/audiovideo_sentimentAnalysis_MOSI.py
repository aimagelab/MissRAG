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
from data.conversation_lib import conv_templates
from data import video_utils
from data.data_utils import make_audio_features
import argparse
from util.misc import get_random_free_port, str2bool
import pandas as pd


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

class MOSI(Dataset):
    def __init__(self, 
                 mode: str = "test",
                 root: str = "/path/to/MOSI") -> None:
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
        video_id = self.data.iloc[index]['video_id']
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
        
        video = load_video(video_path)
        audio = load_audio(audio_path)

        sample = {
            'id': id,
            'video': video,
            'audio': audio,
            'video_id': video_id,
            'clip_id': clip_id,
            'text': text,
            'label': label,
            'annotation': annotation
        }

        return sample

def audio_visual_generate(images, 
                            modal, 
                            task_modals, 
                            task, 
                            text, 
                            conv_template_name, 
                            user_command, 
                            prompt_inside, 
                            missing_prompt, 
                            use_text_modality=True,
                            oneshot=False):
    for i in range(len(images)):
        images[i] = images[i].cuda().to(target_dtype)

    prompts = []
    if task == 'classification':
        for i, inp in enumerate(text):
            if use_text_modality:
                text_i_p = """Input text: {text}. """.format(text=text[i])
                UC = text_i_p + user_command         
            else:
                UC = user_command
                 
            conv = conv_templates[conv_template_name].copy() 
            
            if missing_prompt: # PE
                if prompt_inside: # PE inside last human instruction
                    miss_prompt = conv.get_miss_prompt_inside(prompt=UC, modal=modal, task_modals=task_modals, use_text_modality=use_text_modality)            
                    conv.append_message(conv.roles[0], miss_prompt)
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt())
                else: # PE inside conversational template
                    if conv_template_name == "v1":
                        miss_prompt = conv.get_miss_prompt2(modal=modal, task_modals=task_modals, use_text_modality=use_text_modality) 
                    elif conv_template_name == "v1_audio_video_text": 
                        miss_prompt = conv.get_miss_prompt(modal=modal, task_modals=task_modals, use_text_modality=use_text_modality)
                    else:
                        raise NotImplementedError        
                    conv.append_message(conv.roles[0], UC)
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt(prompt=miss_prompt))
            else:    
                conv.append_message(conv.roles[0], UC)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)

    elif task == 'regression':  
        for i, inp in enumerate(text):  
            if use_text_modality:
                text_i_p = """Dialogue: {text}. """.format(text=text[i])
                UC = text_i_p + user_command         
            else:
                UC = user_command     
            conv = conv_templates[conv_template_name].copy() 
            if missing_prompt: 
                if prompt_inside: 
                    miss_prompt = conv.get_miss_prompt_inside(prompt=UC, modal=modal, task_modals=task_modals, use_text_modality=use_text_modality)            
                    conv.append_message(conv.roles[0], miss_prompt)
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt())
                else: 
                    if conv_template_name == "v1":
                        miss_prompt = conv.get_miss_prompt2(modal=modal, task_modals=task_modals, use_text_modality=use_text_modality) 
                    elif conv_template_name == "v1_audio_video_text": 
                        miss_prompt = conv.get_miss_prompt(modal=modal, task_modals=task_modals, use_text_modality=use_text_modality)
                    else:
                        raise NotImplementedError        
                    conv.append_message(conv.roles[0], UC)
                    conv.append_message(conv.roles[1], None)
                    prompts.append(conv.get_prompt(prompt=miss_prompt))
            else:    
                conv.append_message(conv.roles[0], UC)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
    else:
        raise ValueError(f"Task {task} not supported")
    
    for prompt in prompts:
        print(f"Prompt:{prompt}", flush=True)
        print(f"Length Prompt:{len(prompt)}", flush=True)
    
    with torch.cuda.amp.autocast(dtype=target_dtype):
        if oneshot:
            if len(prompts) != 1:
                raise ValueError(f"Only one prompt is allowed in one-shot setting. Got {len(prompts)} prompts")
            responses = model.generate_multimodal_oneshot(prompts=prompts, images=images, max_gen_len=128, temperature=0.1, top_p=0.75, modal=modal, allowed_token_ids=[6374, 21104, 8178])
        else:
            responses = model.generate_multimodal(prompts=prompts, images=images, max_gen_len=128, temperature=0.1, top_p=0.75, modal=modal)
    
        outputs = []
        for response, prompt in zip(responses, prompts):
            response = response[len(prompt):].split('###')[0]
            response = response.strip()
            outputs.append(response)
    return outputs
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/path/to/MissRAG/OneLLM/pretrained/consolidated.00-of-01.pth"
    )
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSI"
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=[]
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['audio', 'video', 'text']
    )
    parser.add_argument(
        "--use_text_modality",
        action="store_true",
        help="Enable text input modality."
    )
    parser.add_argument(
        "--oneshot", 
        action="store_true",
        help=(
            "Use one-shot generation mode. Instead of autoregressive decoding, the model predicts only a single next token among the allowed token list"
        )
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--user_command_classification",
        type=str,
        default="""Given the class set ["Positive", "Neutral", "Negative"] and considering the emotional tone, facial expressions, and dialogue, what is the sentiment of this video?.""",
        help='Template for the prompt'
    )
    parser.add_argument(
        "--user_command_regression",
        type=str,
        default="""Considering the emotional tone, facial expressions, and dialogue in this video, rate its overall sentiment as a real number between -3.0 (extremely negative) and 3.0 (extremely positive).""",
        help='Template for the prompt'
    ) 
    parser.add_argument(
        "--prompt_inside", action="store_true", help="PE inside last human instruction"
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    parser.add_argument(
        "--missing_prompt",
        action="store_true",
        help="Use PE technique (missing prompt)."
    ) 
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video_text", help="Conversation template to use"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results/eval_sentimentAnalysis_MOSI.json", help="Path to save the answer"
    )
    args = parser.parse_args()  
    
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
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

    if not args.debug:  
        print("Loading pretrained weights ...")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    result = {}
    print("Starting...")
    dataset = MOSI(root=args.root) 
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    predictions = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            id = data['id']
            video = data['video']
            audio = data['audio']
            video_id = data['video_id']
            clip_id = data['clip_id']
            text = data['text']
            label = data['label']
            annotation = data['annotation']

            images=[]
            if 'audio' in args.modal:
                images.append(audio)
            if 'video' in args.modal:
                images.append(video)

            preds_classification = audio_visual_generate(images=images, 
                                        modal=args.modal, 
                                        task_modals=args.task_modals, 
                                        task='classification', 
                                        text=text, 
                                        conv_template_name=args.conversation_template, 
                                        user_command=args.user_command_classification, 
                                        prompt_inside=args.prompt_inside, 
                                        missing_prompt=args.missing_prompt, 
                                        use_text_modality=args.use_text_modality,
                                        oneshot=args.oneshot)

            for video_id, result_classification in zip(id, preds_classification):
                predictions.append({
                    'image_id': video_id,
                    'classification': result_classification.strip(),   
                })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)