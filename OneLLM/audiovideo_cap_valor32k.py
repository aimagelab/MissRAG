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
from util.misc import get_random_free_port


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

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
    video, audio, video_id = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)

    return video, audio, video_id


class CaptionDataset(Dataset):
    def __init__(self, data_path, root) -> None:
        super().__init__()
        self.datas = json.load(open(data_path))
        self.root = root
    
    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, _, _ = parse_video_id(data['video_id'])
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        image_path = f"{self.root}/{video_id}/{video_name}"
        audio_path = f"{self.root}/{video_id}/{audio_name}"

        if not os.path.isfile(image_path):
            print(f"Warning: File {image_path} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        
        if not os.path.isfile(audio_path):
            print(f"Warning: File {audio_path} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        try:
            image = load_video(image_path)
            audio = load_audio(audio_path)
        except:
            print(f"Unable to open {image_path}")
            return None

        return image, audio, data['video_id']


def audio_visual_generate(images, inps, modal, task_modals, conv_template, prompt_inside, missing_prompt):
    for i in range(len(images)):
        images[i] = images[i].cuda().to(target_dtype)

    prompts = []
    for inp in inps:
        conv = conv_templates[conv_template].copy()  
        
        if missing_prompt: # PE
            if prompt_inside: # PE inside last human instruction
                miss_prompt = conv.get_miss_prompt_inside(prompt=inp, modal=modal, task_modals=task_modals)            
                
                conv.append_message(conv.roles[0], miss_prompt)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())

            else: # PE inside conversational template
                if conv_template == "v1":
                    miss_prompt = conv.get_miss_prompt2(modal=modal, task_modals=task_modals) 
                elif conv_template == "v1_audio_video": 
                    miss_prompt = conv.get_miss_prompt(modal=modal, task_modals=task_modals)
                else:
                    raise NotImplementedError    
                      
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt(prompt=miss_prompt))
        else:            
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompts.append(conv.get_prompt())
    
    
    print(f"Prompt:{prompts[0]}", flush=True)

    with torch.cuda.amp.autocast(dtype=target_dtype):
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
        "--data_path", type=str, default='/path/to/VALOR-32K/VALOR-32K-annotations/valor-32k-annotations/desc_test.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/VALOR-32K/data'
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--prompt_inside", type=bool, default=False
    )
    parser.add_argument(
        "--missing_prompt", type=bool, default=False
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default='Provide a detailed description for the given video in one sentence.',
        help='Template for the prompt}'
    )
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video", help="Conversation template to use"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results/eval_valor32k.json", help="Path to save the answer"
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
       
    print("Loading pretrained weights ...")
    checkpoint = torch.load(args.pretrained_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint, strict=False)
    print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    result = {}
    print("Starting...")
    dataset = CaptionDataset(data_path=args.data_path, root=args.root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=custom_collate_fn)
    
    with torch.no_grad():
        outputs = []
        for video, audio, video_ids in tqdm(dataloader):
            images=[]
            if 'video' in args.modal: 
                images.append(video)
            if 'audio' in args.modal:
                images.append(audio)            
            
            prompt = args.prompt_template
            bsz = video.shape[0]
            prompts=[prompt] * bsz
  
            results = audio_visual_generate(images=images, inps=prompts, modal=args.modal, task_modals=args.task_modals, conv_template=args.conversation_template, prompt_inside=args.prompt_inside, missing_prompt=args.missing_prompt)

            for video_id, result in zip(video_ids, results):
                outputs.append({
                    'image_id': video_id,
                    'caption': result.strip()
                })

    with open(args.answer_path, 'w') as f:
        json.dump(outputs, f, indent=4)