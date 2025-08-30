#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import re
import torch
import numpy as np
import h5py

dataset_prefix = "music_avqa" # set the name of the correct dataset ("music_avqa / charadesego / valor")
ids_name = "video" 
split = "train" # set split to "train" or "test" 
root = "/path/to/MissRAG/ImageBind" # root to the current directory
data_dir = f"{root}/IB_embeddings/{split}/{dataset_prefix}_SF" # directory of your .pt files
print(data_dir)
pattern = fr'{dataset_prefix}_video_IB_embeddings_(\d+)\.pt'

video_IB_embeddings_files = []
indices = []
all_files = os.listdir(data_dir)

for filename in all_files:
    match = re.match(pattern, filename)
    if match:
        index = int(match.group(1))
        indices.append(index)
        video_IB_embeddings_files.append((index, filename))

# Sort the files based on indices
video_IB_embeddings_files.sort(key=lambda x: x[0])

# Initialize lists to collect tensors and IDs
all_video_IB_embeddings = []
all_audio_IB_embeddings = []
all_video_ids = []

for index, video_token_filename in video_IB_embeddings_files:
    audio_token_filename = f"{dataset_prefix}_audio_IB_embeddings_{index}.pt"
    video_ids_filename = f"{dataset_prefix}_video_ids_{index}.pt"
    
    # Load tensors
    video_IB_embeddings_np = torch.load(os.path.join(data_dir, video_token_filename))
    audio_IB_embeddings_np = torch.load(os.path.join(data_dir, audio_token_filename))
    video_ids = torch.load(os.path.join(data_dir, video_ids_filename))
    
    # Convert to tensors if needed
    video_IB_embeddings = torch.from_numpy(video_IB_embeddings_np) if isinstance(video_IB_embeddings_np, np.ndarray) else video_IB_embeddings_np
    audio_IB_embeddings = torch.from_numpy(audio_IB_embeddings_np) if isinstance(audio_IB_embeddings_np, np.ndarray) else audio_IB_embeddings_np
    if video_IB_embeddings.shape[0]!=6:
        print(video_IB_embeddings.shape)
        print(video_ids)

    # Collect data
    all_video_IB_embeddings.append(video_IB_embeddings)
    all_audio_IB_embeddings.append(audio_IB_embeddings)
    all_video_ids.extend(video_ids)

# Optionally, concatenate tensors
all_video_IB_embeddings = torch.cat(all_video_IB_embeddings, dim=0)
all_audio_IB_embeddings = torch.cat(all_audio_IB_embeddings, dim=0)

print("Loaded all tensors successfully.")
print(f"Total video IB_embeddings shape: {all_video_IB_embeddings.shape}")
print(f"Total audio IB_embeddings shape: {all_audio_IB_embeddings.shape}")
print(f"Total number of video IDs: {len(all_video_ids)}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_samples = len(all_video_ids)
path = f'{root}/data/IB_embeddings/{split}/IB_embeddings_{split}_{dataset_prefix}.h5'
print(path)
with h5py.File(path, 'w') as h5f:
        # Create datasets with the shape (num_samples, 30, 4096)
        video_ds = h5f.create_dataset(
            'video',
            shape=(num_samples, 1024),
            dtype='float16',
            compression='gzip'
        )
        audio_ds = h5f.create_dataset(
            'audio',
            shape=(num_samples, 1024),
            dtype='float16',
            compression='gzip'
        )
        # Optionally store video_ids if needed
        video_ids_ds = h5f.create_dataset(
            'ids',
            shape=(num_samples,),
            dtype=h5py.string_dtype(encoding='utf-8'), #for charades/valor
            #dtype='int64',
            compression='gzip'
        )
        index = 0
        video_ds[:, :] = all_video_IB_embeddings.numpy()
        audio_ds[:, :] = all_audio_IB_embeddings.numpy()
        video_ids_ds[:] = np.array(all_video_ids)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        