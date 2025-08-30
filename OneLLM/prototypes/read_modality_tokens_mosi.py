#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
import re
import torch
import numpy as np
import h5py

dataset_prefix = "MOSI" # "MOSI" / "MOSEI"
data_dir = f"prototypes/data/modality_tokens/train/{dataset_prefix}_SF" # your directory with .pt files
ids_name = "video" # video question
pattern = fr'{dataset_prefix}_video_tokens_(\d+)\.pt'

# Initialize lists to collect data
video_token_files = []
indices = []

# List all files in the directory
all_files = os.listdir(data_dir)

match_count = 0
for filename in all_files:
    match = re.match(pattern, filename)
    if match:
        print(f"Matched: {filename}")
        match_count += 1
        index = int(match.group(1))
        indices.append(index)
        video_token_files.append((index, filename))

print(f"Found {match_count} files.")
# Sort the files based on indices
video_token_files.sort(key=lambda x: x[0])

# Initialize lists to collect tensors and IDs
all_video_tokens = []
all_audio_tokens = []
all_text = []
all_video_ids = []


for index, video_token_filename in video_token_files:
    audio_token_filename = f"{dataset_prefix}_audio_tokens_{index}.pt"
    text_filename = f"{dataset_prefix}_text_{index}.pt"
    video_ids_filename = f"{dataset_prefix}_ids_{index}.pt"
    
    # # Load tensors
    video_tokens_np = torch.load(os.path.join(data_dir, video_token_filename))
    audio_tokens_np = torch.load(os.path.join(data_dir, audio_token_filename))
    text = torch.load(os.path.join(data_dir, text_filename))
    video_ids = torch.load(os.path.join(data_dir, video_ids_filename))
    
    # # Convert to tensors if needed
    video_tokens = torch.from_numpy(video_tokens_np) if isinstance(video_tokens_np, np.ndarray) else video_tokens_np
    audio_tokens = torch.from_numpy(audio_tokens_np) if isinstance(audio_tokens_np, np.ndarray) else audio_tokens_np
    
    # # Collect data
    all_video_tokens.append(video_tokens)
    all_audio_tokens.append(audio_tokens)
    all_text.extend(text)
    all_video_ids.extend(video_ids)

# Optionally, concatenate tensors
all_video_tokens = torch.cat(all_video_tokens, dim=0)
all_audio_tokens = torch.cat(all_audio_tokens, dim=0)

print("Loaded all tensors successfully.")
print(f"Total video tokens shape: {all_video_tokens.shape}")
print(f"Total audio tokens shape: {all_audio_tokens.shape}")
print(f"Total texts: {len(all_text)}")
print(f"Total number of video IDs: {len(all_video_ids)}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_samples = len(all_video_ids)
with h5py.File(f'prototypes/data/modality_tokens_train_{dataset_prefix}.h5', 'w') as h5f:
        # Create datasets with the shape (num_samples, 30, 4096)
        video_ds = h5f.create_dataset(
            'video',
            shape=(num_samples, 30, 4096),
            dtype='float16',
            compression='gzip'
        )
        audio_ds = h5f.create_dataset(
            'audio',
            shape=(num_samples, 30, 4096),
            dtype='float16',
            compression='gzip'
        )
        # Optionally store video_ids if needed
        text_ds = h5f.create_dataset(
            'text',
            shape=(num_samples,),
            dtype=h5py.string_dtype(encoding='utf-8'),
            # dtype='int64',
            compression='gzip'
        )
        video_ids_ds = h5f.create_dataset(
            'ids',
            shape=(num_samples,),
            dtype=h5py.string_dtype(encoding='utf-8'),
            # dtype='int64',
            compression='gzip'
        )
        index = 0
        video_ds[:, :, :] = all_video_tokens
        audio_ds[:, :, :] = all_audio_tokens
        text_ds[:] = np.array(all_text)
        video_ids_ds[:] = np.array(all_video_ids)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        