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
                 root: str = "/path/to/MOSI",
                 test_IB_path: str = None) -> None:
        super().__init__()
        self.mode = mode
        self.data = pd.read_csv(root+'/labels.csv')
        self.data['id'] = self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)
        self.data = self.data[self.data['mode'] == self.mode]
        self.root = root
        self.video_prefix = 'Raw'
        self.audio_prefix = 'Raw_audio'

        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_text = h5f['text'][:]
            self.IB_ids = h5f['ids'][:]

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

        IB_index = np.where(self.IB_ids == id.encode())[0]
        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]
        IB_text = self.IB_text[IB_index]

        sample = {
            'id': id,
            'video': video,
            'audio': audio,
            'video_id': video_id,
            'clip_id': clip_id,
            'text': text,
            'label': label,
            'annotation': annotation,
            'IB_video': IB_video.flatten(),
            'IB_audio': IB_audio.flatten(),
            'IB_text': IB_text.flatten(),
        }

        return sample

def audio_visual_generate(images,  
                            modal, 
                            task_modals, 
                            text,
                            conversation_template, 
                            user_command,
                            retrieved_tokens, 
                            prototipe_prompt,
                            use_text_modality=True,
                            task = 'classification',
                            oneshot=False):
    for i in range(len(images)):
        images[i] = images[i].cuda().to(target_dtype)

    prompts_llm = []
    for i, inp in enumerate(text):
        text_i_p = """Input text: {text}. """.format(text=text[i])
        UC = text_i_p + user_command         

        conv = conv_templates[conversation_template].copy()  

        if prototipe_prompt: 
            prot_prompt = conv.get_prototipe_prompt(modal=modal, task_modals=task_modals, use_text_modality=use_text_modality, input_text=True)  
            conv.append_message(conv.roles[0], UC) 
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt(prompt=prot_prompt) 
            prompts_llm.append(prompts) 
        else:
            conv.append_message(conv.roles[0], UC)
            conv.append_message(conv.roles[1], None)
            prompts = conv.get_prompt()
            prompts_llm.append(prompts) 

    for prompt in prompts_llm:
        print(f"Prompt:{prompt}", flush=True)
        print(f"Length Prompt:{len(prompt)}", flush=True)

    with torch.cuda.amp.autocast(dtype=target_dtype):
        if oneshot:
            responses = model.generate_multimodal_retrieve_oneshot(prompts=prompts_llm, 
                                                            images=images, 
                                                            max_gen_len=128, 
                                                            temperature=0.1, 
                                                            top_p=0.75, 
                                                            modal=task_modals, 
                                                            retrieved_tokens=retrieved_tokens,
                                                            allowed_token_ids=[6374, 21104, 8178])
        else:
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
                        query1,
                        query2, 
                        k, 
                        train_tokens, 
                        test_batch_size, 
                        n_tokens=30, 
                        d=4096,
                        prototipe_aggregator=None):
    D1, I1 = index_file.search(query1, k)
    if query2 is None:
        D, I = D1, I1
    else:
        D2, I2 = index_file.search(query2, k)
        D = np.concatenate((D1, D2), axis=1)  # Shape: (6, k)
        I = np.concatenate((I1, I2), axis=1)  # Shape: (6, k)
    # For each sample, find the position of the maximum similarity score
    max_positions = np.argpartition(-D, kth=k-1, axis=1)[:, :k] 
    # Retrieve the corresponding indices using advanced indexing
    best_indices = np.take_along_axis(I, max_positions, axis=1) #(b, k)

    if prototipe_aggregator == 'random' and k>1:
        # Generate random indices (one per sample) among the `k` selected
        random_indices = np.random.randint(0, k, size=test_batch_size) 
        best_indices = best_indices[np.arange(test_batch_size), random_indices]  
        top_k = train_tokens[best_indices]
        top_k = top_k.reshape(test_batch_size, n_tokens, d)

    elif prototipe_aggregator=='mean' or k==1:
        top_k = train_tokens[best_indices]           
        top_k = top_k.reshape(test_batch_size, k, n_tokens, d)
        top_k = top_k.mean(axis=1)
    else:
        raise NotImplementedError
    return top_k

def retrieve_text_modality(
                        index_file, # text_index_file
                        query1,
                        query2, 
                        k, 
                        train_tokens, 
                        test_batch_size, 
                        prototipe_aggregator=None):
    D1, I1 = index_file.search(query1, k)

    if query2 is None:
        D, I = D1, I1
    else:
        D2, I2 = index_file.search(query2, k)
        D = np.concatenate((D1, D2), axis=1)  # Shape: (6, k)
        I = np.concatenate((I1, I2), axis=1)  # Shape: (6, k)

    # For each sample, find the position of the maximum similarity score
    max_positions = np.argpartition(-D, kth=k-1, axis=1)[:, :k]
    # Retrieve the corresponding indices using advanced indexing
    best_indices = np.take_along_axis(I, max_positions, axis=1)

    if prototipe_aggregator == 'random' and k>1:
        # Generate random indices (one per sample) among the `k` selected
        random_indices = np.random.randint(0, k, size=test_batch_size) 
        best_indices = best_indices[np.arange(test_batch_size), random_indices]  
        best_indices = np.expand_dims(best_indices, axis=1)

    top_k = train_tokens[best_indices] #(b, k)
    concatenated_list = [b' '.join(row).decode('utf-8') for row in top_k]
    return concatenated_list
    
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
        "--task_modals", nargs='+', type=str, default=['audio', 'video']
    )
    parser.add_argument(
        "--train_modality_tokens_path",        
        type=str,
        default='/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens_train_MOSI.h5',
        help= 'Path to the train modality tokens'
    )
    parser.add_argument(
        "--test_IB_embeddings_path", 
        type=str, 
        default='/path/to/MissRAG/ImageBind/IB_embeddings/test/IB_embeddings_test_mosi.h5'
    )
    parser.add_argument(
        "--train_IB_embeddings_path",        
        type=str,
        default='/path/to/MissRAG/ImageBind/IB_embeddings/train/IB_embeddings_train_mosi.h5',
        help= 'Path to the train IB embeddings'
    )
    parser.add_argument(
        "--k", type=int, default=5
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--user_command_classification",
        type=str,
        default="""Given the class set ["Positive", "Neutral", "Negative"] What is the sentiment of this video?""",
        help='Template for the prompt'
    )
    parser.add_argument(
        "--user_command_regression",
        type=str,
        default="""Considering the emotional tone, facial expressions, and dialogue in this video, rate its overall sentiment as a real number between -3.0 (extremely negative) and 3.0 (extremely positive).""",
        help='Template for the prompt'
    ) 
    parser.add_argument(
        "--prototipe_prompt", 
        action="store_true", 
        help="use PE technique (missing prompt)"
    )
    parser.add_argument(
        "--prototipe_aggregator", type=str, default="mean", help="how k retrieved modality tokens are aggregated"
    )
    parser.add_argument(
        "--oneshot", 
        action="store_true",
        help=(
            "Use one-shot generation mode. Instead of autoregressive decoding, the model predicts only a single next token among the allowed token list"
        )
    )
    parser.add_argument(
        "--conversation_template", type=str, default="v1_audio_video_text", help="Conversation template to use"
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results/eval_sentimentAnalysis_MOSI_retrieval.json", help="Path to save the answer"
    )
    parser.add_argument(
        "--debug", action="store_true",
    )
    parser.add_argument(
        "--use_text_modality",
        action="store_true",
        help="Enable text input modality."
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
    
    if args.debug is False:
        print("Loading pretrained weights ...")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
    model.half().cuda()
    model.eval()
    print(f"Model = {str(model)}")

    print("Loading train modality tokens ...")
    limit = 100
    if args.debug:
        with h5py.File(args.train_modality_tokens_path, 'r') as h5f:
            train_video_tokens = h5f['video'][:limit]  # Shape: (batch_size, 30, 4096)
            train_audio_tokens = h5f['audio'][:limit]  # Shape: (batch_size, 30, 4096)
            train_text_tokens =  h5f['text'][:limit]  # Shape: (batch_size, 30, 4096)
            train_ids          = h5f['ids'][:limit]    # Shape: (batch_size,)
    else:
        with h5py.File(args.train_modality_tokens_path, 'r') as h5f:
            train_video_tokens = h5f['video'][:]  # Shape: (batch_size, 30, 4096)
            train_audio_tokens = h5f['audio'][:]  # Shape: (batch_size, 30, 4096)
            train_text_tokens =  h5f['text'][:]  # Shape: (batch_size, 30, 4096)
            train_ids          = h5f['ids'][:]    # Shape: (batch_size,)
    train_batch_size = train_video_tokens.shape[0]
    print("train_video_tokens: ", train_video_tokens.shape)
    print("train_audio_tokens: ", train_audio_tokens.shape)
    print("train_text_tokens: ", train_text_tokens.shape)
    print("train_ids: ", train_ids.shape)

    print("Loading train IB embeddings ...")
    if args.debug:
        with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:
            train_IB_audio = h5f['audio'][:limit]
            train_IB_video = h5f['video'][:limit]
            train_IB_text = h5f['text'][:limit]
            train_IB_ids = h5f['ids'][:limit]
    else:
        with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:
            train_IB_audio = h5f['audio'][:]
            train_IB_video = h5f['video'][:]
            train_IB_text = h5f['text'][:]
            train_IB_ids = h5f['ids'][:]
    d_IB = train_IB_audio.shape[1]
    IB_index_video = faiss.IndexFlatIP(d_IB)
    IB_index_video.add(train_IB_video)
    IB_index_audio = faiss.IndexFlatIP(d_IB)
    IB_index_audio.add(train_IB_audio)
    IB_index_text = faiss.IndexFlatIP(d_IB)
    IB_index_text.add(train_IB_text)

    assert train_IB_ids.shape==train_ids.shape

    result = {}
    print("Starting...")
    dataset = MOSI(root=args.root, test_IB_path=args.test_IB_embeddings_path) 
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    predictions = []

    recover_modal = list(set(args.task_modals) - set(args.modal))
    if args.use_text_modality is False:
        recover_modal.append("text")
    if len(recover_modal) == 3:
        raise ValueError("At lest one modality between audio, video and text must be present.\nPlease check the --modal and --task_modals arguments. \nmodal: {args.modal}\ntask_modals: {args.task_modals}")
    print(f"The following modalities will be recovered: {recover_modal}")

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
            IB_video = data['IB_video']
            IB_audio = data['IB_audio']
            IB_text = data['IB_text']
    
            images=[]
            retrieved_tokens = {}

            images.append(audio)
            images.append(video)

            # SINGLE MISSING MODALITY CASES
            if 'video' in recover_modal and len(recover_modal) == 1:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(),
                                        IB_text.cpu().numpy(), 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0], 
                                        30,
                                        4096,                                       
                                        args.prototipe_aggregator)
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda().to(target_dtype)
                retrieved_tokens['video'] = retrieved_video_tokens
            
            if 'audio' in recover_modal and len(recover_modal) == 1:
                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_video.cpu().numpy(),
                                        IB_text.cpu().numpy(),  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0], 
                                        30, 
                                        4096,
                                        args.prototipe_aggregator)
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda().to(target_dtype)              
                retrieved_tokens['audio'] = retrieved_audio_tokens
            
            if 'text' in recover_modal and len(recover_modal) == 1:
                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_video.cpu().numpy(),
                                        IB_audio.cpu().numpy(),  
                                        args.k, 
                                        train_text_tokens, 
                                        args.prototipe_aggregator)
                text = retrieved_texts 
            
            # MULTIPLE MISSING MODALITY CASES
            if 'video' in recover_modal and 'audio' in recover_modal and len(recover_modal) == 2:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_text.cpu().numpy(),
                                        None, 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0], 
                                        30,
                                        4096,
                                        args.prototipe_aggregator                                       
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda().to(target_dtype)
                retrieved_tokens['video'] = retrieved_video_tokens

                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_text.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0],  
                                        30, 
                                        4096,
                                        args.prototipe_aggregator)
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda().to(target_dtype)              
                retrieved_tokens['audio'] = retrieved_audio_tokens

            if 'video' in recover_modal and 'text' in recover_modal and len(recover_modal) == 2:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(),
                                        None, 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0], 
                                        30,
                                        4096,
                                        args.prototipe_aggregator                                       
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda().to(target_dtype)
                retrieved_tokens['video'] = retrieved_video_tokens

                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_audio.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_text_tokens, 
                                        args.prototipe_aggregator)
                text = retrieved_texts                  

            if 'audio' in recover_modal and 'text' in recover_modal and len(recover_modal) == 2:
                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_video.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0], 
                                        30, 
                                        4096,
                                        args.prototipe_aggregator)
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda().to(target_dtype)              
                retrieved_tokens['audio'] = retrieved_audio_tokens

                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_video.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_text_tokens, 
                                        args.prototipe_aggregator)
                text = retrieved_texts             

            if args.debug is False:                
                preds = audio_visual_generate(images=images, 
                                        modal=args.modal, 
                                        task_modals=args.task_modals, 
                                        text=text, 
                                        conversation_template=args.conversation_template, 
                                        user_command=args.user_command_classification,  
                                        prototipe_prompt=args.prototipe_prompt, 
                                        retrieved_tokens=retrieved_tokens,
                                        use_text_modality=args.use_text_modality,
                                        oneshot=args.oneshot)
            else:
                preds = ["This is a fake pred" for _ in range(args.batch_size)]

            for video_id, result in zip(id, preds):
                predictions.append({
                    'image_id': video_id,
                    'classification': result.strip()
                })

    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)