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
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Eval CharadesEGO Retrieval")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "av"], help='', required=False, default='v'
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/Charades/CharadesEgo/CharadesEgo_v1_test_only3rd.csv'
    )
    parser.add_argument(
        "--video_path", type=str, default='/path/to/Charades/CharadesEgo_v1'
    )
    parser.add_argument(
        "--audio_path", type=str, default='/path/to/Charades/CharadesEgo_v1_Audio_Extracted'
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/eval_charades_ego_retrieval.json"
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
    def __init__(self, processor, test_path, video_path, audio_path, test_IB_path) -> None:
        super().__init__()
        self.video_path = video_path
        self.audio_path = audio_path 
        self.datas=[]
        with open(test_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.datas.append(row['id'])

        self.processor = processor
        
        with h5py.File(test_IB_path, 'r') as h5f:
            self.IB_audio = h5f['audio'][:]
            self.IB_video = h5f['video'][:]
            self.IB_ids = h5f['ids'][:] 

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        video_id = self.datas[index]
        video_name = video_id + '.mp4'

        vpath = f"{self.video_path}/{video_name}"
        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception
        
        audio_video_tensor = self.processor(vpath, va=True)

        IB_index = np.where(self.IB_ids == video_id.encode())[0]
        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]      

        return audio_video_tensor['video'], audio_video_tensor['audio'], video_id, IB_video.flatten(), IB_audio.flatten()
       
    def get_audio_or_videos(self, ids, modality):
        output_tensor = []
        for id in ids.squeeze(0).tolist(): # for each k
            video_id = id.decode('utf-8')
            video_name = video_id + '.mp4'
            vpath = f"{self.video_path}/{video_name}"

            if not os.path.isfile(vpath):
                print("File {} does not exist".format(vpath))
                raise Exception
            
            audio_video_tensor = self.processor(vpath, va=True)
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
    dataset = CaptionDataset(test_path=args.data_path, video_path=args.video_path, audio_path=args.audio_path, test_IB_path=args.test_IB_embeddings_path, processor=processor['video'])   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    
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
                video_retrieved = dataset.get_audio_or_videos(recovered_video_ids, modality='video')   
            
            if args.modal_type == 'v':
                recovered_audio_ids = retrieve_modality(
                                            IB_index_audio, 
                                            IB_video.cpu().numpy(), 
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read audio from retrieved question_id
                audio_retrieved = dataset.get_audio_or_videos(recovered_audio_ids, modality='audio')                    

            prompts = []
            if args.prototype_prompt:
                prot_prompt = get_prototipe_prompt(modal_type=args.modal_type, task_modals=args.task_modals, input_text=False) 
            else:
                prot_prompt = ""
            
            prompt = args.prompt_template
            if args.modal_type == 'av':
                sample = {'video': video[0], 'audio': audio[0]}   
            elif args.modal_type == 'v':
                sample = {'video': video[0], 'audio': audio_retrieved}   
            elif args.modal_type == 'a':
                sample = {'video': video_retrieved, 'audio': audio[0]}   
            else:
                raise NotImplementedError
            
            result = mm_infer(
                image_or_video=sample,
                instruct=prompt,
                model=model,
                tokenizer=tokenizer,
                modal="video",
                do_sample=False,
                prot_prompt=prot_prompt
            )
            print(f"Answer: {result}")
            predictions.append({'image_id': video_ids[0], 'caption': result})
            
            # Save every 1000 predictions
            count += 1
            if count % 1000 == 0:
                print(f"Saving predictions at {count}")
                answer_path = args.answer_path.replace('.json', f'_{count}.json')
                with open(answer_path, 'w') as f:
                    json.dump(predictions, f)
                    
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)