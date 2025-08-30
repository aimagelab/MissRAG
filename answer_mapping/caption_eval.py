from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

# VALOR32K
results_file = '/path/to/MissRAG/results/eval_valor32k_retrieval_M_video_k1.json' # your result file
test_file = '/path/to/VALOR-32K/VALOR-32K-annotations/valor-32k-annotations/desc_test.json' 

# creo coco dataset
with open(test_file, 'r') as f:
    dataset = json.load(f)
with open(results_file , 'r') as f:
    result = json.load(f)
ids=[]
for res in result:
    ids.append(res['image_id'])

# Transform to COCO-style format
coco_format = {
    'images': [{'id': entry['video_id'], 'file_name': entry['video_id']} for i, entry in enumerate(dataset) if entry['video_id'] in ids],
    'annotations': [{'id': entry['video_id'], 'image_id': entry['video_id'], 'caption': entry['desc']} for i, entry in enumerate(dataset) if entry['video_id'] in ids],
    'info': {'year': 2024, 'version': '1.0', 'description': 'This is a 1.0 version of the VALOR32K dataset'},
}

# create coco object and coco_result object
coco = COCO(coco_format)
coco_result = coco.loadRes(results_file)

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.5f}')
