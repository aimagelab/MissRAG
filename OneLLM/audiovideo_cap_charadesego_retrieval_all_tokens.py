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
import csv


def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

class CaptionDataset(Dataset):
    def __init__(self, test_path, video_path, audio_path, test_IB_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path    
        self.datas=[]
        with open(test_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.datas.append(row['id'])

        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_ids = h5f['ids'][:]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        video_id = self.datas[index]
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'

        video_path = f"{self.video_path}/{video_name}"
        audio_path = f"{self.audio_path}/{audio_name}"

        if not os.path.isfile(video_path):
            print(f"Warning: File {video_path} is missing.")
            raise Exception
        
        if not os.path.isfile(audio_path):
            print(f"Warning: File {audio_path} is missing.")
            raise Exception
        
        video = load_video(video_path)
        audio = load_audio(audio_path)

        IB_index = np.where(self.IB_ids == video_id.encode())[0]
        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]

        return video, audio, video_id, IB_video.flatten(), IB_audio.flatten()

def audio_visual_generate(images, 
                            inps, 
                            modal, 
                            task_modals, 
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
            conv.append_message(conv.roles[0], inp) 
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt(prompt=prot_prompt) 
            prompts_llm.append(prompts) 
       
        else:
            conv.append_message(conv.roles[0], inp)
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
        "--test_path", type=str, default='/path/to/Charades/CharadesEgo/CharadesEgo_v1_test_only3rd.csv'
    )
    parser.add_argument(
        "--video_path", type=str, default='/path/to/Charades/CharadesEgo_v1'
    )
    parser.add_argument(
        "--audio_path", type=str, default='/path/to/Charades/CharadesEgo_v1_Audio_Extracted'
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
        default='/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_charadesego.h5',
        help= 'Path to the train modality tokens'
    )
    parser.add_argument(
        "--test_IB_embeddings_path", 
        type=str, 
        default='/path/to/MissRAG/ImageBind/IB_embeddings/test/IB_embeddings_test_charadesego.h5'
    )
    parser.add_argument(
        "--train_IB_embeddings_path",        
        type=str,
        default='/path/to/MissRAG/ImageBind/IB_embeddings/train/IB_embeddings_train_charadesego.h5',
        help= 'Path to the train IB embeddings'
    )
    parser.add_argument(
        "--k", type=int, default=5
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--prototipe_prompt", type=bool, default=False
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default='Provide a detailed description for the given video in one sentence.',
        help='Template for the prompt'
    )
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video", help="Conversation template to use"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAGù/results/eval_charadesego_retrieval.json", help="Path to save the answer"
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
    other_limit = 1000
    other_modality_tokens_path = [
        "/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_MOSEI.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_MOSI.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_valor.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_music_avqa.h5",
    ]
    other_video_modality_tokens = []
    other_audio_modality_tokens = []
    if args.debug:
        with h5py.File(args.train_modality_tokens_path, 'r') as h5f:        
            train_video_tokens = h5f['video'][:limit]
            train_audio_tokens = h5f['audio'][:limit]
            train_ids          = h5f['ids'][:limit]
        for path in other_modality_tokens_path:
            with h5py.File(path, 'r') as h5f:
                other_video_modality_tokens.append(h5f['video'][:other_limit])
                other_audio_modality_tokens.append(h5f['audio'][:other_limit])
    else:
        with h5py.File(args.train_modality_tokens_path, 'r') as h5f:
            train_video_tokens = h5f['video'][:]  # Shape: (batch_size, 30, 4096)
            train_audio_tokens = h5f['audio'][:]  # Shape: (batch_size, 30, 4096)
            train_ids          = h5f['ids'][:]    # Shape: (batch_size,)
        for path in other_modality_tokens_path:
            with h5py.File(path, 'r') as h5f:
                other_video_modality_tokens.append(h5f['video'][:])
                other_audio_modality_tokens.append(h5f['audio'][:])
    
    train_video_tokens = np.concatenate([train_video_tokens, *other_video_modality_tokens])
    train_audio_tokens = np.concatenate([train_audio_tokens, *other_audio_modality_tokens])
    print("train_video_tokens: ", train_video_tokens.shape)
    print("train_audio_tokens: ", train_audio_tokens.shape)
    print("train_ids: ", train_ids.shape)

    print("Loading train IB embeddings ...")
    other_IB_embeddings_path = [
        "/path/to/MissRAG/OneLLM/prototypes/data/IB_embeddings/train/IB_embeddings_train_mosei.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/IB_embeddings/train/IB_embeddings_train_mosi.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/IB_embeddings/train/IB_embeddings_train_valor.h5",
        "/path/to/MissRAG/OneLLM/prototypes/data/IB_embeddings/train/IB_embeddings_train_music_avqa.h5"
    ]
    other_video_IB_embeddings = []
    other_audio_IB_embeddings = []
    
    if args.debug:
        with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:        
            train_IB_audio = h5f['audio'][:limit]
            train_IB_video = h5f['video'][:limit]
            train_IB_ids = h5f['ids'][:limit]
        for path in other_IB_embeddings_path:
            with h5py.File(path, 'r') as h5f:
                other_video_IB_embeddings.append(h5f['video'][:other_limit])
                other_audio_IB_embeddings.append(h5f['audio'][:other_limit])
    
    else:
        with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:   
            train_IB_audio = h5f['audio'][:]
            train_IB_video = h5f['video'][:]
            train_IB_ids = h5f['ids'][:]
        for path in other_IB_embeddings_path:
            with h5py.File(path, 'r') as h5f:
                other_video_IB_embeddings.append(h5f['video'][:other_limit])
                other_audio_IB_embeddings.append(h5f['audio'][:other_limit])

    train_IB_video = np.concatenate([train_IB_video, *other_video_IB_embeddings])
    train_IB_audio = np.concatenate([train_IB_audio, *other_audio_IB_embeddings])  
    d_IB = train_IB_audio.shape[1]
    IB_index_video = faiss.IndexFlatIP(d_IB)
    IB_index_video.add(train_IB_video)
    IB_index_audio = faiss.IndexFlatIP(d_IB)
    IB_index_audio.add(train_IB_audio)

    assert train_IB_ids.shape==train_ids.shape

    result = {}
    print("Starting...")
    dataset = CaptionDataset(test_path=args.test_path, video_path=args.video_path, audio_path=args.audio_path, test_IB_path=args.test_IB_embeddings_path)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    predictions = []
    recover_modal = set(args.task_modals) - set(args.modal)
    recover_modal = list(recover_modal)[0]
    correct = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, video_ids, IB_video, IB_audio = data
            images=[]
            retrieved_tokens = {}

            images.append(video)
            images.append(audio)

            prompt = args.prompt_template
            bsz = video.shape[0]
            prompts=[prompt] * bsz

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
               
            preds = audio_visual_generate(images=images, 
                                    inps=prompts, 
                                    modal=args.modal, 
                                    task_modals=args.task_modals, 
                                    conversation_template=args.conversation_template,  
                                    prototipe_prompt=args.prototipe_prompt, 
                                    retrieved_tokens=retrieved_tokens)
            
            for video_id, result in zip(video_ids, preds):
                predictions.append({
                    'image_id': video_id,
                    'caption': result.strip()
                })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)