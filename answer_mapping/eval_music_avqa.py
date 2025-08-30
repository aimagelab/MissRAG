#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
sys.path.append('./')
import os
import json
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datas = json.load(open('results/music_avqa_MV.json'))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def map_number_to_string(number_str):
    number_map = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten"
    }
    return number_map.get(number_str, number_str)

# Example usage
input_str = "1"
mapped_str = map_number_to_string(input_str)
print(f"Mapped string: {mapped_str}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
run_correct = 0
run_total = 0
for data in datas:
    run_total += 1
    answer = data['answer']
    pred = data['gt_answer']
    answer = map_number_to_string(answer)
    pred = pred.strip().lower()
    answer = answer.strip().lower()
    if (pred in answer) or (answer in pred):
        run_correct += 1
    

print(f"Run accuracy: {float(run_correct)/run_total * 100}")
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%