import numpy as np
import json
ank = np.array([0.56233161, 0.24250873, 0.0578919 , 0.06932502, 0.04003936, 0.02206968, 0.04275707, 0.06578732])

# Load data from .jsonl files
def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Save data to .jsonl files
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

test_path = 'NIKL_EA_2023/nikluge-ea-2023-test.jsonl'
test_data = load_jsonl(test_path)

# Ensemble results
res = np.stack([np.array([np.load(f'results_feat_ens_{f}_{hdim}.npy') for f in range(5)]
    ).mean(axis=0) for hdim in [64, 96, 128, 256, 512, 768, 1024, 1280]]).mean(axis=0)>0.5
output_name = 'output.jsonl'

emotions = ['joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']
for i in range(len(test_data)):
    i_dict = {}
    for j, k in enumerate(emotions) :
        if res[i,j] == 1:
            i_dict[k] = 'True'
        else:
            i_dict[k] = 'False'
    test_data[i]['output'] = i_dict
save_jsonl(test_data, output_name)

