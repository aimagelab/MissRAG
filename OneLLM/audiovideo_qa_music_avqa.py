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
import ast
import re


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

class AVQADataset(Dataset):
    def __init__(self, data_path, root) -> None:
        super().__init__()
        self.root = root 
        self.datas = json.load(open(data_path))
        #self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}

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
        
        image = load_video(image_path)
        audio = load_audio(audio_path)

        question_id = data['question_id']
        question = data['question_content']
        answer = data['anser']
        templ_values = data['templ_values']
        templ_list = ast.literal_eval(templ_values)

        # Find all the placeholders
        placeholders = re.findall(r"<(.*?)>", question)

        if len(placeholders) != len(templ_list):
            raise ValueError("The number of placeholders does not match the number of templ_values.")

        for placeholder, value in zip(placeholders, templ_list):
            question = question.replace(f"<{placeholder}>", value)
        return image, audio, question, question_id, answer


def audio_visual_generate(images, inps, modal, task_modals, prompt_template, conversation_template, prompt_inside, missing_prompt):
    for i in range(len(images)):
        images[i] = images[i].cuda().to(target_dtype)

    prompts_llm = []
    for inp in inps:
        conv = conv_templates[conversation_template].copy()  
        if missing_prompt: # PE
            if prompt_inside: # PE inside last human instruction
                miss_prompt = inp + "\n" + conv.get_miss_prompt_inside(prompt=prompt_template, modal=modal, task_modals=task_modals) 
                
                conv.append_message(conv.roles[0], miss_prompt) 
                conv.append_message(conv.roles[1], None)
                prompts = conv.get_prompt()
                prompts_llm.append(prompts) 
            else: # PE inside conversational template
                if conversation_template == "v1":
                    miss_prompt = conv.get_miss_prompt2(modal=modal, task_modals=task_modals) 
                elif conversation_template == "v1_audio_video": 
                    miss_prompt = conv.get_miss_prompt(modal=modal, task_modals=task_modals, compensation_strategy=args.compensation_strategy) 
                else:
                    raise NotImplementedError   
                prompt = inp + "\n" + prompt_template 
                
                conv.append_message(conv.roles[0], prompt) 
                conv.append_message(conv.roles[1], None)
                prompts = conv.get_prompt(prompt=miss_prompt) 
                prompts_llm.append(prompts) 
        else:
            prompt = inp + "\n" + prompt_template 
            
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt()
            prompts_llm.append(prompts) 

    for prompt in prompts_llm:
        print(prompt)

    with torch.cuda.amp.autocast(dtype=target_dtype):
        responses = model.generate_multimodal(prompts=prompts_llm, images=images, max_gen_len=128, temperature=0.1, top_p=0.75, modal=modal)
        outputs = []
        for response, prompt in zip(responses, prompts_llm):
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
        "--data_path", type=str, default='/path/to/MUSIC_AVQA_git/MUSIC-AVQA/data/json/avqa-test_corrected.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/MUSIC-AVQA'
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results/eval_music_avqa.json"
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
        "--compensation_strategy",  action='store_false',   # sets the value to False when the flag is used
            dest='feature_enabled', # name of the variable
            default=True,           # default value when the flag is not used
            help='Disable the feature (enabled by default)'
    )
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video", help="Conversation template to use"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default='Answer the question using a single word or phrase.',
        help='Template for the prompt'
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
    dataset = AVQADataset(data_path=args.data_path, root=args.root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    predictions = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, questions, question_ids, answers = data
            images=[]
            
            if 'video' in args.modal:
                images.append(video)
            if 'audio' in args.modal:
                images.append(audio)
        
            preds = audio_visual_generate(images=images, inps=questions, modal=args.modal, task_modals=args.task_modals, prompt_template=args.prompt_template, conversation_template=args.conversation_template, prompt_inside=args.prompt_inside, missing_prompt=args.missing_prompt)

            for question, pred, question_id, answer in zip(questions, preds, question_ids, answers):
                predictions.append({'question_id': question_id.item(), 'answer': pred, 'gt_answer': answer}) 

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)