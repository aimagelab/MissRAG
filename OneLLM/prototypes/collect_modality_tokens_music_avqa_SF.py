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

def load_video(video_path):
    video_feats = video_utils.load_and_transform_video_data(video_path, video_path, clip_duration=1, clips_per_video=5)
    return video_feats[:, :, 0]

def load_audio(audio_path):
    fbank = make_audio_features(audio_path, mel_bins=128)
    fbank = fbank.transpose(0, 1)[None]     #[1, 128, 1024]
    return fbank

class AVQADataset(Dataset):
    def __init__(self, data_path, root) -> None:
        super().__init__()
        self.root = root 
        self.datas = json.load(open(data_path))
        #self.id_to_video_ids = {i: self.datas[i]['video_id'] for i in range(len(self.datas))}

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
        video_id = data['video_id']
        return image, audio, question_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="/path/to/MissRAG/OneLLM/pretrained/consolidated.00-of-01.pth"
    )
    parser.add_argument(
        "--data_path", type=str, default='/path/to/MUSIC_AVQA_git/MUSIC-AVQA/data/json/avqa-train.json'
    )
    parser.add_argument(
        "--root", type=str, default='/path/to/MUSIC-AVQA'
    )
    parser.add_argument(
        "--modal", nargs='+', type=str, default=['video', 'audio']
    )
    parser.add_argument(
        "--batch_size", type=int, default=6
    )
    parser.add_argument(
        "--answer_path", type=str, default="/path/to/MissRAG/OneLLM/prototypes/data/modality_tokens/train/music_avqa_SF"
    )
    parser.add_argument("--debug", action='store_true', help="debug, don't use model but fake data")
    args = parser.parse_args()  
    
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

    if args.debug is False:
        with default_tensor_type(dtype=target_dtype, device="cuda"):
            model = MetaModel("onellm", "config/llama2/7B.json", None, "config/llama2/tokenizer.model")
        print("Loading pretrained weights ...")
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=False)
        print("load result:\n", msg)
        model.half().cuda()
        model.eval()
        print(f"Model = {str(model)}")

    dataset = AVQADataset(data_path=args.data_path, root=args.root)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            drop_last=False)
    num_samples = len(dataset)
    if args.debug:
        num_samples = 6
    batch_size = args.batch_size
    index = 0
    
    for video, audio, video_ids in tqdm(dataloader):
        print(video.shape, audio.shape, video_ids)
        
        # Move data to the appropriate device if using GPUs
        video = video.cuda().to(target_dtype)
        audio = audio.cuda().to(target_dtype)
        
        # Compute modality tokens
        if args.debug:
            video_modality_tokens = torch.randn(video.shape[0], 30, 4096)
            audio_modality_tokens = torch.randn(audio.shape[0], 30, 4096)
        else:
            with torch.inference_mode():
                video_modality_tokens = model.llma.encode_image(video, modal="video")
                audio_modality_tokens = model.llma.encode_image(audio, modal="audio")
        print(video_modality_tokens.shape, audio_modality_tokens.shape, video_ids)
        
        # Move tensors to CPU and convert to NumPy arrays
        video_tokens_np = video_modality_tokens.cpu().numpy()
        audio_tokens_np = audio_modality_tokens.cpu().numpy()
        
        batch_size_actual = video_tokens_np.shape[0]     
        
        for i in range(batch_size_actual):
            print(video_ids[i], video_tokens_np[i].shape, audio_tokens_np[i].shape)
            
        torch.save(video_tokens_np, f"{args.answer_path}/music_avqa_video_tokens_{index}.pt")
        torch.save(audio_tokens_np, f"{args.answer_path}/music_avqa_audio_tokens_{index}.pt")
        torch.save(video_ids, f"{args.answer_path}/music_avqa_video_ids_{index}.pt")   
  
        index += batch_size_actual  
  
        if args.debug:
            break
    print("DONE")




 
    

