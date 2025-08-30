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
sys.path.append('./')



def parse_args():
    parser = argparse.ArgumentParser(description="MOSEI Eval")
    parser.add_argument("--model_path", default='/path/to/MissRAG/VideoLLaMA2/weigths', help="path to configuration file.")
    parser.add_argument(
        "--modal_type", choices=["a", "v", "av"], help='', required=False, default='av'
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/MUSIC_AVQA_git/MUSIC-AVQA/data/json/avqa-test_corrected.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/MUSIC-AVQA'
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/VideoLLaMA2/results/eval_music_avqa.json"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--missing_prompt", action='store_true', default=False
    )
    parser.add_argument(
        "--prompt_template", type=str, default=' Answer the question using a single word or phrase.'
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

class AVQADataset(Dataset):
    def __init__(self, data_path, root, processor) -> None:
        super().__init__()
        self.root = root 
        self.datas = json.load(open(data_path))
        self.processor = processor

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = str(data['video_id'])
        video_name = video_id + '.mp4'
        
        vpath = os.path.join(self.root, video_name)

        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception

        audio_video_tensor = self.processor(vpath, va=True) 

        question = data['question_content']
        question_id = data['question_id']
        answer = data['anser']

        templ_values = data['templ_values']
        templ_list = ast.literal_eval(templ_values)

        # Find all the placeholders
        placeholders = re.findall(r"<(.*?)>", question)

        if len(placeholders) != len(templ_list):
            raise ValueError("The number of placeholders does not match the number of templ_values.")

        for placeholder, value in zip(placeholders, templ_list):
            question = question.replace(f"<{placeholder}>", value, 1)

        return {
            "question_id": question_id,
            "video": audio_video_tensor['video'],
            "audio": audio_video_tensor['audio'], 
            "question": question,
            "answers": answer,
        }

 # ========================================

if __name__ == "__main__":
    # ========================================
    #             Model Initialization
    # ========================================

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
    dataset = AVQADataset(data_path=args.data_path, root=args.root, processor=processor['video'])   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    count = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, questions, answers, question_ids = data["video"], data["audio"], data["question"], data["answers"], data["question_id"]

            if args.modal_type == 'av':
                sample = {'video': video[0], 'audio': audio[0]}   
            elif args.modal_type == 'v':
                sample = video[0]
            elif args.modal_type == 'a':
                sample = audio[0]
            else:
                raise NotImplementedError
            
            question = questions[0] + args.prompt_template
             
            result = mm_infer(
                image_or_video=sample,
                instruct=question,
                model=model,
                tokenizer=tokenizer,
                modal='audio' if args.modal_type == "a" else "video",
                do_sample=False,
            )

            print(f"Answer: {result}")
            predictions.append({'question_id': question_ids[0].item(), 'answer': result, 'gt_answer': answers[0]})
            
            # Save every 1000 predictions
            count += 1
            if count % 1000 == 0:
                print(f"Saving predictions at {count}")
                answer_path = args.answer_path.replace('.json', f'_{count}.json')
                with open(answer_path, 'w') as f:
                    json.dump(predictions, f)
                    
    with open(args.answer_path, 'w') as f:
        json.dump(predictions, f)