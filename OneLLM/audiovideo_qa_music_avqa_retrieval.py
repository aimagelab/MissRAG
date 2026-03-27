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
from typing import List
import faiss
import h5py
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
    def __init__(self, data_path, root, test_IB_path) -> None:
        super().__init__()
        self.root = root 
        self.datas = json.load(open(data_path))
        #self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}
        
        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_ids = h5f['ids'][:]

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

        IB_index = np.where(self.IB_ids == question_id)[0]

        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]

        if IB_video.shape[0] > 1:
            IB_video = IB_video[0]
            print(f"WARNING: at index {index},  IB_video.shape: ", IB_video.shape)
        if IB_audio.shape[0] > 1:
            IB_audio = IB_audio[0]
            print(f"WARNING: at index {index},  IB_audio.shape: ", IB_audio.shape)

        return image, audio, question, question_id, answer, IB_video.flatten(), IB_audio.flatten()


def audio_visual_generate(images, 
                            inps, 
                            modal, 
                            task_modals, 
                            prompt_template, 
                            conversation_template, 
                            retrieved_tokens, 
                            prototipe_prompt):
    for i in range(len(images)):
        images[i] = images[i].cuda().to(target_dtype)

    prompts_llm = []
    for inp in inps:
        conv = conv_templates[conversation_template].copy()  
        
        if prototipe_prompt: 
            prot_prompt = conv.get_prototipe_prompt(modal=modal, task_modals=task_modals)  
            prompt = inp + "\n" + prompt_template + '.'
            conv.append_message(conv.roles[0], prompt) 
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt(prompt=prot_prompt) 
            prompts_llm.append(prompts) 
        
        else:
            prompt = inp + "\n" + prompt_template + '.'
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt()
            prompts_llm.append(prompts) 

    for prompt in prompts_llm:
        print(prompt)

    with torch.cuda.amp.autocast(dtype=target_dtype):
        responses = model.generate_multimodal_retrieve(prompts=prompts_llm, 
                                                            images=images, 
                                                            max_gen_len=128, 
                                                            temperature=0.1, 
                                                            top_p=0.75, 
                                                            modal=task_modals, 
                                                            retrieved_tokens=retrieved_tokens)
        outputs = []
        for response, prompt in zip(responses, prompts_llm):
            response = response[len(prompt):].split('###')[0]
            response = response.strip()
            outputs.append(response)
    return outputs

def retrieve_modality(
                        index_file, 
                        query, 
                        k, 
                        train_tokens, 
                        test_batch_size, 
                        n_tokens=30, 
                        d=4096):
    D, I = index_file.search(query, k)
    top_k = train_tokens[I]
    top_k = top_k.reshape(test_batch_size, k, n_tokens, d)
    top_k = top_k.mean(axis=1)
    return top_k
    
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
        "--modal", nargs='+', type=str, default=['video'] #modals available
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['video', 'audio'] 
    )
    parser.add_argument(
        "--train_modality_tokens_path",        
        type=str,
        default='/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_music_avqa.h5',
        help= 'Path to the train modality tokens'
    )
    parser.add_argument(
        "--test_IB_embeddings_path", 
        type=str, 
        default='/path/to/MissRAG/ImageBind/IB_embeddings/test/IB_embeddings_test_music_avqa.h5'
    )
    parser.add_argument(
        "--train_IB_embeddings_path",        
        type=str,
        default='/path/to/MissRAG/ImageBind/IB_embeddings/train/IB_embeddings_train_music_avqa.h5',
        help= 'Path to the train IB embeddings'
    )
    parser.add_argument(
        "--k", type=int, default=5
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--prototipe_prompt", 
        action="store_true", 
        help="use PE technique (missing prompt)"
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default='Answer the question and explain the reason in one sentence',
        help='Template for the prompt'
    )
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video", help="Conversation template to use"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results/eval_qa_music_avqa_retrieval.json", help="Path to save the answer"
    )
    parser.add_argument("--debug", action='store_true', help="debug, don't use model but fake data")
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
    
    if args.debug is False:
        print("Loading pretrained weights ...")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
        model.half().cuda()
        model.eval()
        print(f"Model = {str(model)}")

    print("Loading train modality tokens ...")
    limit = 1000
    with h5py.File(args.train_modality_tokens_path, 'r') as h5f:
        if args.debug:
            train_video_tokens = h5f['video'][:limit]
            train_audio_tokens = h5f['audio'][:limit]
            train_ids          = h5f['ids'][:limit]
        else:
            train_video_tokens = h5f['video'][:]  # Shape: (batch_size, 30, 4096)
            train_audio_tokens = h5f['audio'][:]  # Shape: (batch_size, 30, 4096)
            train_ids          = h5f['ids'][:]    # Shape: (batch_size,)
    train_batch_size = train_video_tokens.shape[0]
    print("train_video_tokens: ", train_video_tokens.shape)
    print("train_audio_tokens: ", train_audio_tokens.shape)
    print("train_ids: ", train_ids.shape)

    print("Loading train IB embeddings ...")
    with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:
        if args.debug:
            train_IB_audio = h5f['audio'][:limit]
            train_IB_video = h5f['video'][:limit]
            train_IB_ids = h5f['ids'][:limit]
        else:
            train_IB_audio = h5f['audio'][:]
            train_IB_video = h5f['video'][:]
            train_IB_ids = h5f['ids'][:]
    d_IB = train_IB_audio.shape[1]
    IB_index_video = faiss.IndexFlatIP(d_IB)
    IB_index_video.add(train_IB_video)
    IB_index_audio = faiss.IndexFlatIP(d_IB)
    IB_index_audio.add(train_IB_audio)

    assert train_IB_ids.shape==train_ids.shape

    result = {}
    print("Starting...")
    dataset = AVQADataset(data_path=args.data_path, root=args.root, test_IB_path=args.test_IB_embeddings_path)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    predictions = []
    recover_modal = set(args.task_modals) - set(args.modal)
    recover_modal = list(recover_modal)[0]

    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, questions, question_ids, answers, IB_video, IB_audio = data
            images=[]
            retrieved_tokens = {}

            images.append(video)
            images.append(audio)

            if 'video' in recover_modal:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(), 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0], 
                                        30,
                                        4096                                       
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda().to(target_dtype)
                retrieved_tokens['video'] = retrieved_video_tokens
            
            if 'audio' in recover_modal:
                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_video.cpu().numpy(), 
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0], 
                                        30, 
                                        4096)
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda().to(target_dtype)              
                retrieved_tokens['audio'] = retrieved_audio_tokens

            if args.debug is False:                
                preds = audio_visual_generate(images=images, 
                                        inps=questions, 
                                        modal=args.modal, 
                                        task_modals=args.task_modals, 
                                        prompt_template=args.prompt_template, 
                                        conversation_template=args.conversation_template,  
                                        prototipe_prompt=args.prototipe_prompt, 
                                        retrieved_tokens=retrieved_tokens)
            else:
                preds = ["This is a fake pred" for _ in range(args.batch_size)]

            for question, pred, question_id, answer in zip(questions, preds, question_ids, answers):
                predictions.append({'question_id': question_id.item(), 'answer': pred, 'gt_answer': answer})

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)