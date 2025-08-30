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
import pandas as pd
sys.path.append('./')
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
import pandas as pd
import copy
sys.path.append('./')
import h5py
import faiss
from videollama2.conversation import get_prototipe_prompt

def parse_args():
    parser = argparse.ArgumentParser(description="MOSI Eval Retrieval")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "t", "av", "at", "vt", "avt"], help='', required=False, default='av'
    )
    parser.add_argument(
        "--task_modals", nargs='+', type=str, default=['audio', 'video']
    )
    parser.add_argument(
        "--root", type=str, default="/path/to/MOSI"
    )
    parser.add_argument(
        "--user_command",
        type=str,
        default="""Given the class set ["Positive", "Neutral", "Negative"], what is the sentiment of this video?.""",
        help='Template for the prompt'
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/MOSI_eval_retrieval.json"
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
    parser.add_argument(
        "--start_idx", type=int, default=0
    )
    parser.add_argument("--debug", action='store_true', help="debug mode on")
    parser.add_argument("--prototype_prompt", action='store_true', default=False)
    args = parser.parse_args()
    return args

def setup_seeds(args):
    seed = args.seed 

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True    

class MOSI(Dataset):
    def __init__(self, 
                 mode: str = "test",
                 root: str = "/path/to/MOSI",
                 test_IB_path: str = None) -> None:
        super().__init__()
        self.mode = mode
        self.data = pd.read_csv(root+'/labels.csv')
        self.data['id'] = self.data['video_id'].astype(str) + '_' + self.data['clip_id'].astype(str)
        self.train_data = copy.deepcopy(self.data)
        self.data = self.data[self.data['mode'] == self.mode]
        self.root = root
        self.video_prefix = 'Raw'
        self.audio_prefix = 'Raw_audio'
        self.processor = processor
        
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
        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception
        
        audio_video_tensor = self.processor(vpath, va=True)
                
        IB_index = np.where(self.IB_ids == id.encode())[0]
        IB_video = self.IB_video[IB_index]
        IB_audio = self.IB_audio[IB_index]
        IB_text = self.IB_text[IB_index]

        sample = {
            'id': id,
            'video': audio_video_tensor['video'],
            'audio': audio_video_tensor['audio'],
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
    
    def get_audio_or_videos_or_text(self, ids):
        videos_list = []
        audios_list = []
        texts = []
        decoded_ids = [id[0].decode('utf-8') for id in ids]
        for id in decoded_ids: # for each k
            video_id, clip_id = id.rsplit('_', 1)
            id = id[::-1].replace('_', '/', 1)[::-1] # replace _ with / to match the file structure
            video_name = id + '.mp4'
            vpath = os.path.join(self.root, self.video_prefix, video_name)
            
            audio_video_tensor = self.processor(vpath, va=True)

            text_row = self.train_data[(self.train_data['video_id'] == video_id)]
            text_row = text_row[(text_row['clip_id'] == int(clip_id))]

            if not text_row.empty:
                text = text_row.iloc[0]['text']
            else:
                raise Exception(f"No text found for video_id {video_id} and clip_id {clip_id}")
            
            audios_list.append(audio_video_tensor['audio'])
            videos_list.append(audio_video_tensor['video'])
            texts.append(text)

        return videos_list, audios_list, texts

def retrieve_modality(
                        index_file, 
                        query1,
                        query2, 
                        k, 
                        train_IB_ids,
                        ): 
    D1, I1 = index_file.search(query1, k)
    if query2 is None:
        D, I = D1, I1
    else:
        D2, I2 = index_file.search(query2, k)
        D = np.concatenate((D1, D2), axis=1)
        I = np.concatenate((I1, I2), axis=1)
    max_positions = np.argpartition(-D, kth=k-1, axis=1)[:, :k]
    best_indices = np.take_along_axis(I, max_positions, axis=1)
    top_k = train_IB_ids[best_indices]
    print("indexes: ", top_k) #(B, k)
    return top_k                   

if __name__ == "__main__":
    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
    setup_seeds(args)

    model, processor, tokenizer = model_init(args.model_path)
    model.eval()
    setup_seeds(args)
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    dataset = MOSI(processor=processor['video'], root=args.root, test_IB_path=args.test_IB_embeddings_path)   
    dataloader = DataLoader(dataset, 
                            batch_size = args.batch_size, 
                            num_workers=args.n_workers, 
                            shuffle=False, 
                            pin_memory=False, 
                            drop_last=False)
    print('Initialization Finished')

    limit = 10
    print("Loading train IB embeddings ...")
    with h5py.File(args.train_IB_embeddings_path, 'r') as h5f:
        if args.debug:
            train_IB_audio = h5f['audio'][:limit]
            train_IB_video = h5f['video'][:limit]
            train_IB_text = h5f['text'][:limit]
            train_IB_ids = h5f['ids'][:limit]
        else:
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

    assert train_IB_ids.shape[0] == train_IB_video.shape[0] == train_IB_audio.shape[0] == train_IB_text.shape[0], "Mismatch in the number of IDs and embeddings"
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    elements = 0
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

            if elements < args.start_idx:
                elements += 1 
                continue
            
            #audio+text
            if args.modal_type == 'at':
                print(IB_audio.cpu().numpy().shape)
                recovered_video_ids = retrieve_modality(
                                        IB_index_video, 
                                        IB_audio.cpu().numpy(), 
                                        IB_text.cpu().numpy(),
                                        args.k, 
                                        train_IB_ids, 
                                        )
                # Read video from retrieved question_id
                video_retrieved, _, _ = dataset.get_audio_or_videos_or_text(recovered_video_ids)   
                video = video_retrieved
            
            #video+text
            elif args.modal_type == 'vt':
                recovered_audio_ids = retrieve_modality(
                                            IB_index_audio, 
                                            IB_video.cpu().numpy(), 
                                            IB_text.cpu().numpy(),
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read audio from retrieved question_id
                _, audio_retrieved, _ = dataset.get_audio_or_videos_or_text(recovered_audio_ids)                    
                audio = audio_retrieved
            
            #video+audio
            elif args.modal_type == 'av':
                recovered_text_ids = retrieve_modality(
                                            IB_index_text, 
                                            IB_video.cpu().numpy(), 
                                            IB_audio.cpu().numpy(),
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read text from retrieved question_id
                _, _, texts_retrieved = dataset.get_audio_or_videos_or_text(recovered_text_ids)
                texts = texts_retrieved
            
            #text
            elif args.modal_type == 't':
                recovered_video_ids = retrieve_modality(
                                            IB_index_video, 
                                            IB_text.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                recovered_audio_ids = retrieve_modality(
                                            IB_index_audio, 
                                            IB_text.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read video and audio from retrieved question_id
                video_retrieved, _, _ = dataset.get_audio_or_videos_or_text(recovered_video_ids)    
                _, audio_retrieved, _ = dataset.get_audio_or_videos_or_text(recovered_audio_ids)
                video = video_retrieved
                audio = audio_retrieved
            
            # audio
            elif args.modal_type == 'a':
                recovered_video_ids = retrieve_modality(
                                            IB_index_video, 
                                            IB_audio.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                recovered_text_ids = retrieve_modality(
                                            IB_index_text, 
                                            IB_audio.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read video and text from retrieved question_id
                video_retrieved, _, _ = dataset.get_audio_or_videos_or_text(recovered_video_ids)    
                _, _, texts_retrieved = dataset.get_audio_or_videos_or_text(recovered_text_ids)
                video = video_retrieved
                texts = texts_retrieved
            
            # video
            elif args.modal_type == 'v':
                recovered_audio_ids = retrieve_modality(
                                            IB_index_audio, 
                                            IB_video.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                recovered_text_ids = retrieve_modality(
                                            IB_index_text, 
                                            IB_video.cpu().numpy(), 
                                            None,
                                            args.k, 
                                            train_IB_ids, 
                                            )
                # Read audio and text from retrieved question_id
                _, audio_retrieved, _ = dataset.get_audio_or_videos_or_text(recovered_audio_ids)    
                _, _, texts_retrieved = dataset.get_audio_or_videos_or_text(recovered_text_ids)   
                audio = audio_retrieved
                texts = texts_retrieved

            sample = {'video': video[0], 'audio': audio[0]} 

            prompts = []
            if args.prototype_prompt:
                prot_prompt = get_prototipe_prompt(modal_type=args.modal_type, task_modals=args.task_modals, input_text=True) 
            else:
                prot_prompt = ""

            for text in texts:
                text_i_p = """Input text: {text}. """.format(text=text)
                question = text_i_p + args.user_command         

            result = mm_infer(
                image_or_video=sample,
                instruct=question,
                model=model,
                tokenizer=tokenizer,
                modal="video",
                do_sample=False,
                prot_prompt=prot_prompt,
            )
            print(f"Answer: {result}")

            for video_id, result_classification in zip(id, [result]):
                predictions.append({
                    'image_id': video_id,
                    'classification': result.strip(),
                })
                    
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f, indent=4)                                    

