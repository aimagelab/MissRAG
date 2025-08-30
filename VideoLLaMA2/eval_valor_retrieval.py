import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm    
import json
import re
import ast
from videollama2 import model_init, mm_infer_batch, mm_infer
import sys
from videollama2.dist_utils import get_rank
import h5py
import faiss
import ast
import re
from videollama2.conversation import get_prototipe_prompt

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
    if len(batch) == 0:
        return None, None, None, None, None
    video, audio, video_id, IB_video, IB_audio = zip(*batch)
    video = torch.stack(video)
    audio = torch.stack(audio)
    IB_video = torch.stack(IB_video)
    IB_audio = torch.stack(IB_audio)

    return video, audio, video_id, IB_video, IB_audio

def parse_args():
    parser = argparse.ArgumentParser(description="Eval Valor Retrieval")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "av"], help='', required=False, default='v'
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
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/eval_valor_retrieval.json"
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
        "--batch_size", type=int, default=1
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
    parser.add_argument("--debug", action='store_true', help="debug mode on")
    parser.add_argument("--prototype_prompt", action='store_true', default=False)
    parser.add_argument(
        "--prompt_template", type=str, default='Provide a detailed description for the given video in one sentence.'
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
    def __init__(self, processor, data_path, root, test_IB_path) -> None:
        super().__init__()
        self.test_data = json.load(open(data_path))
        self.datas = []
        self.root = root

        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_ids = h5f['ids'][:]

        for data in self.test_data:
            if data['video_id'].encode("utf-8") in self.IB_ids:
                self.datas.append(data)   
        self.processor = processor

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id, _, _ = parse_video_id(data['video_id'])

        video_name = video_id + '.mp4'
        vpath = f"{self.root}/{video_id}/{video_name}"

        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            return None

        try:
            audio_video_tensor = self.processor(vpath, va=True) 
        except:
            print(f"Unable to open {vpath}")
            return None
        
        IB_index = np.where(self.IB_ids == data['video_id'].encode())[0]
        IB_video = torch.from_numpy(self.IB_video[IB_index])
        IB_audio = torch.from_numpy(self.IB_audio[IB_index])

        return audio_video_tensor['video'], audio_video_tensor['audio'], data["video_id"], IB_video.flatten(), IB_audio.flatten()
       
    def get_audio_or_videos(self, ids, modality):
        output_tensor = []
        for id in ids.squeeze(0).tolist(): # for each k
            video_id, _, _ = parse_video_id(id.decode('utf-8'))
            video_name = video_id + '.mp4'

            vpath = f"{self.root}/{video_id}/{video_name}"
            
            if not os.path.isfile(vpath):
                print("File {} does not exist".format(vpath))
                return None
            
            try:
                audio_video_tensor = self.processor(vpath, va=True)
            except:
                print(f"Unable to open {vpath}")
                return None
            
            output_tensor.append(audio_video_tensor[modality])
            
        out_tensor = torch.stack(output_tensor, dim=0) #(k, ...)
        return out_tensor.mean(axis=0)
            
def retrieve_modality(
                        index_file, 
                        query, 
                        k, 
                        train_IB_ids): 
    D, I = index_file.search(query, k)
    top_k = train_IB_ids[I]
    print("indexes: ", top_k) #(B, k)
    return top_k    

 # ========================================

if __name__ == "__main__":
    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
        
    model, processor, tokenizer = model_init(args.model_path)
    model.eval()
    setup_seeds(args)
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    dataset = CaptionDataset(data_path=args.data_path, root=args.root, test_IB_path=args.test_IB_embeddings_path, processor=processor['video'])   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False, collate_fn=custom_collate_fn)
    
    limit = 10
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

    #assert train_IB_ids.shape==train_ids.shape   
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    count = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, video_ids, IB_video, IB_audio = data
            # audio: (1, 2998, 128), video: (1, 8, 3, 384, 384)

            samples = {}
            if args.modal_type == 'a':
                print(IB_audio.cpu().numpy().shape)
                recovered_video_ids = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(), 
                                        args.k, 
                                        train_IB_ids, 
                                        )
                # Read video from retrieved question_id
                video = dataset.get_audio_or_videos(recovered_video_ids, modality='video')   
                if video is None:
                    continue
            if args.modal_type == 'v':
                recovered_audio_ids = retrieve_modality(
                                            IB_index_audio, 
                                            IB_video.cpu().numpy(), 
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read audio from retrieved question_id
                audio = dataset.get_audio_or_videos(recovered_audio_ids, modality='audio')  
                if audio is None:
                    continue                  

            prompts = []
            if args.prototype_prompt:
                prot_prompt = get_prototipe_prompt(modal_type=args.modal_type, task_modals=args.task_modals, input_text=False) 
            else:
                prot_prompt = ""
            
            prompt = args.prompt_template
            sample = {'video': video[0], 'audio': audio[0]}               
            #print(prot_prompt)
            
            result = mm_infer(
                image_or_video=sample,
                instruct=prompt,
                model=model,
                tokenizer=tokenizer,
                modal="video",
                do_sample=False,
                prot_prompt=prot_prompt,
            )
            print(f"Answer: {result}")
            predictions.append({'image_id': video_ids[0], 'caption': result})
            
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)