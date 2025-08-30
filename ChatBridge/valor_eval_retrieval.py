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
import h5py
import faiss


def parse_args():
    parser = argparse.ArgumentParser(description="Valor Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--modal", nargs='+', type=str, default=[]
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/VALOR-32K/VALOR-32K-annotations/valor-32k-annotations/desc_test.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/VALOR-32K/data'
    )
    parser.add_argument(
       "--task_modals", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/results_chatbridge/eval_valor_cap.json"
    )
    parser.add_argument(
        "--train_modality_tokens_path", type=str, default="/path/to/MissRAG/ChatBridge/prototypes/data/modality_tokens_train_valor.h5"
    )
    parser.add_argument(
        "--test_IB_embeddings_path", 
        type=str, 
        default='/path/to/MissRAG/ImageBind/IB_embeddings/test/IB_embeddings_test_valor.h5'
    )
    parser.add_argument(
        "--train_IB_embeddings_path",        
        type=str,
        default='/path/to/MissRAG/ImageBind/IB_embeddings/train/IB_embeddings_train_valor.h5',
        help= 'Path to the train IB embeddings'
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--k", type=int, default=1
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
    parser.add_argument("--debug", action='store_true', help="debug mode on")
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
    video, audio, video_id, IB_video, IB_audio = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)
    IB_video = torch.stack(IB_video)
    IB_audio = torch.stack(IB_audio)

    return video, audio, video_id, IB_video, IB_audio

class CaptionDataset(Dataset):
    def __init__(self, data_path, root, test_IB_path, vis_processor, aud_processor) -> None:
        super().__init__()
        self.test_data = json.load(open(data_path))
        self.datas = []
        self.root = root

        self.vis_processor = vis_processor
        self.aud_processor = aud_processor

        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_ids = h5f['ids'][:]

        for data in self.test_data:
            if data['video_id'].encode("utf-8") in self.IB_ids:
                self.datas.append(data)         

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, _, _ = parse_video_id(data['video_id'])
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        vpath = f"{self.root}/{video_id}/{video_name}"
        apath = f"{self.root}/{video_id}/{audio_name}"

        if not os.path.isfile(vpath):
            print(f"Warning: File {vpath} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        
        if not os.path.isfile(apath):
            print(f"Warning: File {apath} is missing. Skipping...")
            return None  # Indicate that this item should be skipped
        try:
            frms = self.vis_processor(vpath)
            auds = self.aud_processor(apath)
        except:
            print(f"Unable to open {vpath}")
            return None
        
        IB_index = np.where(self.IB_ids == data['video_id'].encode())[0]
        IB_video = torch.from_numpy(self.IB_video[IB_index])
        IB_audio = torch.from_numpy(self.IB_audio[IB_index])

        return frms, auds, data['video_id'], IB_video.flatten(), IB_audio.flatten()

def retrieve_modality(
                        index_file, 
                        query, 
                        k, 
                        train_tokens, 
                        test_batch_size, 
                        n_tokens=32, 
                        d=5120):

    D, I = index_file.search(query, k)
    top_k = train_tokens[I]
    top_k = top_k.reshape(test_batch_size, k, n_tokens, d)
    top_k = top_k.mean(axis=1)
    return top_k            
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
    dataset = CaptionDataset(data_path=args.data_path, root=args.root, test_IB_path=args.test_IB_embeddings_path, vis_processor=vis_processor, aud_processor=aud_processor)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
    
    print("Loading train modality tokens ...")
    limit = 10
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
    
    print('Initialization Finished')

    with open(args.prompt_file_path, 'r') as file:
        prompt_file = json.load(file)

    print("Starting...")
    recover_modal = set(args.task_modals) - set(args.modal)
    recover_modal = list(recover_modal)[0]
    print("Recover Modal: ", recover_modal)
    outputs = []
    with torch.no_grad():
        for video, audio, video_ids, IB_video, IB_audio in tqdm(dataloader):
            video = video.permute(0,2,1,3,4)
            samples = {}
            retrieved_tokens = {}

            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()

            if recover_modal == 'video':
                print(IB_audio.cpu().numpy().shape)
                recovered_video = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(), 
                                        args.k, 
                                        train_video_tokens, 
                                        video.shape[0]
                                        )
                retrieved_video_tokens = torch.from_numpy(recovered_video).cuda()
                retrieved_tokens["image"] = retrieved_video_tokens
            
            if recover_modal == 'audio':
                recovered_audio = retrieve_modality(
                                            IB_index_audio, 
                                            IB_video.cpu().numpy(), 
                                            args.k, 
                                            train_audio_tokens, 
                                            video.shape[0]
                                            )
                retrieved_audio_tokens = torch.from_numpy(recovered_audio).cuda()
                retrieved_tokens["audio"] = retrieved_audio_tokens     

            samples["retrieved_tokens"] = retrieved_tokens   
            
            prompts = []

            if args.prototype_prompt:
                prot_prompt =  " " + get_prototipe_prompt(modal=args.modal, task_modals=args.task_modals, input_text=False) 
            else:
                prot_prompt = ""

            bsz = video.shape[0]

            for question in range(bsz):
                conv = CONV_VIDEO.copy()
                prefix = "Given following video: <query> and its background audio: <query>."
                prompt_template = prompt_file["tva-cap"][args.index_prompt]
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
        
            for video_id, result in zip(video_ids, results):
                outputs.append({
                    'image_id': video_id,
                    'caption': result.strip()
                })
                
    with open(args.answer_path, 'w') as f:
        json.dump(outputs, f)