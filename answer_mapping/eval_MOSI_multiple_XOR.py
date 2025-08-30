#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import json
import os
from tqdm import tqdm
import os
import pandas as pd
import numpy as np


root = "/path/to/MOSI"
mode = "test"
data = pd.read_csv(root+'/labels.csv')
data['id'] = data['video_id'].astype(str) + '_' + data['clip_id'].astype(str)
data = data[data['mode'] == mode]
data = data.set_index('id')

MOSI_CLASS_2 = ["Positive", "Negative"]
MOSI_CLASS_3 = ["Positive", "Negative", "Neutral"]
MOSI_CLASS = MOSI_CLASS_3

def get_key_by_value(value, dictionary):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

# dictionary to hold result paths
res_paths = {
    #"Complete": "/path/to/MissRAG/results/eval_MOSEI_V2_C.json",
}

# result order  in the output dataframe
res_order = [
    #"Complete"
]

# result names in the output dataframe
names = [
        #"Complete"
]

XOR_values = []
for res_name in res_order:
    print("\n\n\n")
    res_path = res_paths[res_name]
    print(res_name, ": ", res_path)
    with open(res_path , "r") as fp:
        res = json.load(fp)
    #Prediction of the exact class
    acc=0
    tot=0
    class_list = MOSI_CLASS
    for result in res:
        tot+=1
        label = data.loc[result["image_id"]]['annotation']
        label = label.lower()
        neg_label = "not " + label
        pred = result['classification'].lower()

        count_match = 0
        for gt in MOSI_CLASS:
            gt = gt.lower()
            if gt in pred:
                count_match += 1

        if count_match > 1:
            continue
        else:
            if (pred==label):
                acc += 1
            elif (label in pred) and not(neg_label in pred):
                acc += 1

    acc = acc/tot
    acc = acc*100
    acc = round(acc, 2)
    print(f"Accuracy (predict the exact class): {acc}")
    XOR_values.append(acc)

    

results_df = pd.DataFrame({
    # 'metohd': res_order,
    'names': names,
    'XOR': XOR_values,
})
#results_df.to_csv("results/XOR_not_MOSI_retrieval_results_2.csv", index=False)

print(results_df)
print("Done!")
# %%
