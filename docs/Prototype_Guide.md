# Creation of the Modality Poolings
Our Retrieval-Augmented Generation (RAG) framework needs two pools: a pool of training-set prototypes, used as keys, and a pool of test-set embeddings, used as queries. We use [ImageBind](https://github.com/facebookresearch/ImageBind) to generate the embeddings, as it maps data from different modalities into a unified embedding space where similar data points are close to each other. This section provides a detailed explanation of how to construct the two embedding pools for each dataset.

## 1 - Create the Prototypes
Create the two pools for each dataset you want to test using the following scripts. The resulting embeddings will be saved as .pt files. Please refer to the [Data Structure Guide](Data.md) for more details about the specific structure of each dataset. 

### Music AVQA 
For MusicAVQA, create the two pools by running:
```bash
python collect_IB_embeddings_music_avqa_SF.py
  --data_path <PATH> \               # Path to the train or test annotation file
  --root <ROOT_PATH> \               # Path to the folder with video/audio files
  --answer_path <OUTPUT_PATH> \      # Directory for saving .pt files 
  --batch_size <BATCH_SIZE> 
```

### Valor
For Valor, create the two pools by running:
```bash
python collect_IB_embeddings_valor_SF.py
  --data_path <PATH> \               # Path to the train or test annotation file
  --root <ROOT_PATH> \               # Path to the folder with video/audio files
  --answer_path <OUTPUT_PATH> \      # Directory for saving .pt files 
  --batch_size <BATCH_SIZE>
```

### CharadesEGO
For CharadesEGO, create the two pools by running:

```bash
python collect_IB_embeddings_valor_SF.py
  --data_path <PATH> \               # Path to the train or test annotation file
  --video_path <VIDEO_PATH> \        # Path to the video files
  --audio_path <AUDIO_PATH> \        # Path to the audio files
  --answer_path <OUTPUT_PATH> \      # Directory for saving .pt files 
  --batch_size <BATCH_SIZE>         
```

### MOSI 
For MOSI, create the two pools by running:
```bash
python collect_IB_embeddings_MOSI_SF.py
  --root <PATH> \                   # Path to the folder with the dataset files
  --mode <MODE> \                   # "train" or "test"
  --answer_path <OUTPUT_PATH> \     # Directory for saving .pt files 
  --batch_size <BATCH_SIZE> 
```

### MOSEI
For MOSEI, create the two pools by running:
```bash
python collect_IB_embeddings_MOSEI_SF.py
  --root <PATH> \                   # Path to the folder with the dataset files
  --mode <MODE> \                   # "train" or "test"
  --answer_path <OUTPUT_PATH> \     # Directory for saving .pt files 
  --batch_size <BATCH_SIZE> 
```

### 2 - Create the `.h5` files
Create the `.h5` files of Music AVQA, Valor and CharadesEGO by running:
```bash
python read_IB_embeddings.py
```
and the `.h5` files of MOSI and MOSEI by running:
```bash
python read_IB_embeddings_mosi.py
```
setting in the python files the correct dataset, the correct split (train or test) and the `data_dir` parameter which correspond to the `answer_path` previously used for the creation of the dataset pool. 
