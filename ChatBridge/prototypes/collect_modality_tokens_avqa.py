
import sys
sys.path.append('./')
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
from chatbridge.processors.blip_processors import BlipAudioEvalProcessor, BlipQuestionProcessor
from chatbridge.processors.alpro_processors import AlproVideoEvalProcessor
from chatbridge.models.chatbridge_only_modality_encoders import ChatBridge_OME
from chatbridge.common.registry import registry
from tqdm import tqdm    
import json
from omegaconf import OmegaConf
import re
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="AVQA Eval")
    parser.add_argument("--cfg_path", help="path to configuration file.", default="/path/to/MissRAG/ChatBridge/eval_configs/chatbridge_eval.yaml")
    #arser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/MUSIC-AVQA'
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/MUSIC_AVQA_git/MUSIC-AVQA/data/json/avqa-train.json'
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/ChatBridge/prototypes/data/modality_tokens/train/music_vqa_SF"
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--n_workers", type=int, default=4
    )
    #parser.add_argument(
    #    "--missing_prompt", type=bool, default=False
    #)
    parser.add_argument(
        "--prompt_template", type=str, default='Based on the video and audio, could you provide a short answer to the question:'
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    args = parser.parse_args()
    return args

def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

class AVQADataset(Dataset):
    def __init__(self, vis_processor, aud_processor, text_processor, data_path, root) -> None:
        super().__init__()
        self.datas = json.load(open(data_path))
        #self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}
        self.vis_processor = vis_processor
        self.aud_processor = aud_processor
        self.text_processor = text_processor
        self.root = root

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        data = self.datas[index]
        video_id = str(data['video_id'])
        video_name = video_id + '.mp4'
        audio_name = video_id + '.mp3'
        vpath = os.path.join(self.root, video_name)
        apath = os.path.join(self.root, audio_name)

        if not os.path.isfile(apath):
            print("File {} does not exist".format(apath))
            raise Exception
        
        if not os.path.isfile(vpath):
            print("File {} does not exist".format(vpath))
            raise Exception
        
        frms = self.vis_processor(vpath)
        auds = self.aud_processor(apath)
        question = self.text_processor(data['question_content'])

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
            "video": frms,
            "audio": auds, 
            "question": question,
            "answers": answer,
        }

 # ========================================

if __name__ == "__main__":
    print('Initializing Model')
    args = parse_args()
    os.makedirs(os.path.dirname(args.answer_path), exist_ok=True) 
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True) 
        
    random.seed(args.seed)

    #cfg = Config(args)
    config = OmegaConf.load(args.cfg_path)
    setup_seeds(config)
    #model_config.device_8bit = args.gpu_id
    # model_cls = registry.get_model_class(config.model.arch)
    model = ChatBridge_OME.from_config(config.model)
    model.cuda()
    model.eval()
    print('Initialization Finished')

    print('Initializing Processor and Dataset')
    vis_processor = AlproVideoEvalProcessor(image_size=224, n_frms=4)
    aud_processor = BlipAudioEvalProcessor()
    text_processor = BlipQuestionProcessor()

    dataset = AVQADataset(vis_processor=vis_processor, aud_processor=aud_processor, text_processor=text_processor, data_path=args.data_path, root=args.root)   
    dataloader = DataLoader(dataset, batch_size = args.batch_size, num_workers=args.n_workers, shuffle=False, pin_memory=True, drop_last=False)
    print('Initialization Finished')

    print("Starting...")
    predictions = []
    correct = 0
    index = 0
    with torch.no_grad():
        for data in tqdm(dataloader):
            video, audio, questions, answers, question_ids = data["video"], data["audio"], data["question"], data["answers"], data["question_id"]
            video = video.permute(0,2,1,3,4)
            samples = {}
            if 'video' in args.modal:
                samples["image"] = video.cuda()
            if 'audio' in args.modal:
                samples["audio"] = audio.cuda()

            image_tok = model.encode_img(samples["image"])[0]
            audio_tok = model.encode_audio(samples["audio"])[0]        

            image_tok_np = image_tok.cpu().numpy()
            audio_tok_np = audio_tok.cpu().numpy()

            batch_size_actual = image_tok_np.shape[0]

            for i in range(batch_size_actual):
                print(question_ids[i], image_tok_np[i].shape, audio_tok_np[i].shape)

            torch.save(image_tok_np, f"{args.answer_path}/music_avqa_video_tokens_{index}.pt")
            torch.save(audio_tok_np, f"{args.answer_path}/music_avqa_audio_tokens_{index}.pt")
            torch.save(question_ids, f"{args.answer_path}/music_avqa_video_ids_{index}.pt")

            index += batch_size_actual

    print("DONE")