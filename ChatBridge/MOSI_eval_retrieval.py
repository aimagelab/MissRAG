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
from chatbridge.conversation.conversation_lib import get_prototipe_prompt
from chatbridge.processors.blip_processors import BlipAudioEvalProcessor, BlipQuestionProcessor
from chatbridge.processors.alpro_processors import AlproVideoEvalProcessor
from chatbridge.common.registry import registry
from tqdm import tqdm    
import json
from omegaconf import OmegaConf
import pandas as pd
import h5py
import faiss


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
       "--task_modals", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--use_text_modality", action='store_true', default=False, help='Enable or disable the text input modality. Accepts true/false.'
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results_chatbridge/eval_music_avqa.json"
    )
    parser.add_argument(
        "--train_modality_tokens_path",        
        type=str,
        default='/path/to/MissRAG/ChatBridge/prototypes/data/modality_tokens_train_MOSI.h5',
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
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=4
    )
    parser.add_argument("--debug", action='store_true', help="debug, don't use model but fake data")
    parser.add_argument("--prototype_prompt", action='store_true', default=False)
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

class MOSI(Dataset):
    def __init__(self, vis_processor, aud_processor, test_IB_path, text_processor=None, mode: str = "test", root: str = "/path/to/MOSI") -> None:
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

        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_text = h5f['text'][:]
            self.IB_ids = h5f['ids'][:]

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
        
        IB_index = np.where(self.IB_ids == id.encode())[0]
        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]
        IB_text = self.IB_text[IB_index]

        sample = {
            'id': id,
            'video': frms,
            'audio': auds,
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


def retrieve_modality(
                        index_file, 
                        query1,
                        query2, 
                        k, 
                        train_tokens, 
                        test_batch_size, 
                        n_tokens=32, 
                        d=5120):
    query1 = np.ascontiguousarray(query1, dtype=np.float32)
    D1, I1 = index_file.search(query1, k)
    if query2 is None:
        D, I = D1, I1
    else:
        query2 = np.ascontiguousarray(query2, dtype=np.float32)
        D2, I2 = index_file.search(query2, k)
        D = np.concatenate((D1, D2), axis=1)  # Shape: (6, k)
        I = np.concatenate((I1, I2), axis=1)  # Shape: (6, k)

    # For each sample, find the position of the maximum similarity score
    max_positions = np.argpartition(-D, kth=k-1, axis=1)[:, :k]
    # Retrieve the corresponding indices using advanced indexing
    best_indices = np.take_along_axis(I, max_positions, axis=1)
    top_k = train_tokens[best_indices]
    top_k = top_k.reshape(test_batch_size, k, n_tokens, d)
    top_k = top_k.mean(axis=1)
    return top_k

def retrieve_text_modality(
                        index_file, # text_index_file
                        query1,
                        query2, 
                        k, 
                        train_tokens):
    
    query1 = np.ascontiguousarray(query1, dtype=np.float32)
    D1, I1 = index_file.search(query1, k)
    if query2 is None:
        D, I = D1, I1
    else:
        query2 = np.ascontiguousarray(query2, dtype=np.float32)
        D2, I2 = index_file.search(query2, k)
        D = np.concatenate((D1, D2), axis=1)  # Shape: (6, k)
        I = np.concatenate((I1, I2), axis=1)  # Shape: (6, k)

    # For each sample, find the position of the maximum similarity score
    max_positions = np.argpartition(-D, kth=k-1, axis=1)[:, :k]
    # Retrieve the corresponding indices using advanced indexing
    best_indices = np.take_along_axis(I, max_positions, axis=1)
    top_k = train_tokens[best_indices]
    concatenated_list = [b' '.join(row).decode('utf-8') for row in top_k]
    return concatenated_list
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

    dataset = MOSI(root=args.root, test_IB_path=args.test_IB_embeddings_path, vis_processor=vis_processor, aud_processor=aud_processor)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    
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

    train_IB_video = np.ascontiguousarray(train_IB_video, dtype=np.float32)
    train_IB_audio = np.ascontiguousarray(train_IB_audio, dtype=np.float32)
    train_IB_text  = np.ascontiguousarray(train_IB_text, dtype=np.float32)

    IB_index_video = faiss.IndexFlatIP(d_IB)
    IB_index_video.add(train_IB_video)
    IB_index_audio = faiss.IndexFlatIP(d_IB)
    IB_index_audio.add(train_IB_audio)
    IB_index_text = faiss.IndexFlatIP(d_IB)
    IB_index_text.add(train_IB_text)

    assert train_IB_ids.shape==train_ids.shape    
    
    print('Initialization Finished')

    with open(args.prompt_file_path, 'r') as file:
        prompt_file = json.load(file)

    print("Starting...")
    recover_modal = list(set(args.task_modals) - set(args.modal))
    if args.use_text_modality is False:
        recover_modal.append("text")
    if len(recover_modal) == 3:
        raise ValueError("At lest one modality between audio, video and text must be present.\nPlease check the --modal and --task_modals arguments. \nmodal: {args.modal}\ntask_modals: {args.task_modals}")
    print(f"The following modalities will be recovered: {recover_modal}")
    
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
            IB_video = data['IB_video']
            IB_audio = data['IB_audio']
            IB_text = data['IB_text']
            print(f"id:{id}")

            video = video.permute(0,2,1,3,4)
            samples = {}
            
            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()

            retrieved_tokens = {}
            if 'video' in recover_modal and len(recover_modal) == 1:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(),
                                        IB_text.cpu().numpy(), 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0]                                   
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda()
                retrieved_tokens['image'] = retrieved_video_tokens
            
            elif 'audio' in recover_modal and len(recover_modal) == 1:
                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_video.cpu().numpy(),
                                        IB_text.cpu().numpy(),  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0]
                                        )
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda()              
                retrieved_tokens['audio'] = retrieved_audio_tokens

            elif 'text' in recover_modal and len(recover_modal) == 1:
                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_video.cpu().numpy(),
                                        IB_audio.cpu().numpy(),  
                                        args.k, 
                                        train_text_tokens
                                )
                texts = retrieved_texts 

            # MULTIPLE MISSING MODALITY CASES
            elif 'video' in recover_modal and 'audio' in recover_modal and len(recover_modal) == 2:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_text.cpu().numpy(),
                                        None, 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0]
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda()
                retrieved_tokens['image'] = retrieved_video_tokens

                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_text.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0]
                                        )
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda()              
                retrieved_tokens['audio'] = retrieved_audio_tokens

            elif 'video' in recover_modal and 'text' in recover_modal and len(recover_modal) == 2:
                retrieved_video_tokens = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(),
                                        None, 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0]
                                        )
                retrieved_video_tokens = torch.from_numpy(retrieved_video_tokens).cuda()
                retrieved_tokens['image'] = retrieved_video_tokens

                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_audio.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_text_tokens
                                        )
                texts = retrieved_texts                  

            elif 'audio' in recover_modal and 'text' in recover_modal and len(recover_modal) == 2:
                retrieved_audio_tokens = retrieve_modality(
                                        IB_index_audio, 
                                        IB_video.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_audio_tokens, 
                                        video.shape[0]
                                        )
                retrieved_audio_tokens = torch.from_numpy(retrieved_audio_tokens).cuda()              
                retrieved_tokens['audio'] = retrieved_audio_tokens

                retrieved_texts = retrieve_text_modality(
                                        IB_index_text, 
                                        IB_video.cpu().numpy(),
                                        None,  
                                        args.k, 
                                        train_text_tokens
                                        )
                texts = retrieved_texts  
            else:
                raise ValueError("At least one modality between audio, video and text must be present.\nPlease check the --modal and --task_modals arguments. \nmodal: {args.modal}\ntask_modals: {args.task_modals}")   
            
            samples["retrieved_tokens"] = retrieved_tokens  
            prompts = []

            if args.prototype_prompt:
                prot_prompt =  " " + get_prototipe_prompt(modal=args.modal, task_modals=args.task_modals, use_text_modality=args.use_text_modality, input_text=True) 
            else:
                prot_prompt = ""
                
            for text in texts:
                conv = CONV_VIDEO.copy()
                prefix = "Given following video: <query>, its background audio: <query> and the input text: <TEXT>.".replace('<TEXT>', text)
                prompt_template = random.choice(prompt_file["tva-sa"])
                if len(prompts)==0:
                    samples['task'] = 'tva'

                conv.append_message(conv.roles[0], prefix+prot_prompt)  
                conv.append_message(conv.roles[0], prompt_template)
                conv.append_message(conv.roles[1], None)
                prompts.append(conv.get_prompt())
            
            for prompt in prompts:
                print(f"Prompt:{prompt}\n", flush=True)
            
            samples['conversation'] = prompts
             
            results = model.forward_inference_retrieval(samples)
        
            for video_id, result in zip(id, results): 
                predictions.append({
                    'image_id': video_id,
                    'classification': result.strip(),    
                })
                
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)