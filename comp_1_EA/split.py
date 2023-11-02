import json
import numpy as np
import random
from sklearn.model_selection import KFold
from skmultilearn.model_selection import iterative_train_test_split


# Set a fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load data from .jsonl files
def load_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Save data to .jsonl files
def save_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Load data
data_path = 'NIKL_EA_2023'
train_data = load_jsonl(f"{data_path}/nikluge-ea-2023-train.jsonl")
dev_data = load_jsonl(f"{data_path}/nikluge-ea-2023-dev.jsonl")

# Combine data
combined_data = train_data + dev_data
#ipdb> combined_data[0]['output']
#{'joy': 'True', 'anticipation': 'False', 'trust': 'False', 'surprise': 'False', 'disgust': 'False', 'fear': 'False', 'anger': 'False', 'sadness': 'False'}
labels = np.array([[int(item['output'][emotion]=='True') for emotion in [
    'joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']] for item in combined_data])

K = 5  # or any desired value for K splits

# Since `iterative_train_test_split` only splits into 2 parts at once, we'll use a loop for K parts
current_data = combined_data
current_labels = labels

# Storing indices of the data instead of data itself
indices = np.arange(len(combined_data))

k_fold_splits = []
for _ in range(K - 1):  # K-1 splits will leave us with 1 last fold
    # Splitting a portion of the data iteratively based on indices
    train_indices, train_labels, test_indices, test_labels = iterative_train_test_split(
                    indices[:, np.newaxis], current_labels, test_size=1/(K-_))
    print(train_labels.mean(axis=0), test_labels.mean(axis=0))
    print(len(train_labels), len(test_labels))
    train_indices, test_indices = np.hstack(train_indices), np.hstack(test_indices)

    # Fetching the actual data based on indices
    X_train = [combined_data[i] for i in train_indices]
    X_test = [combined_data[i] for i in test_indices]
    # Add the current split to the list
    k_fold_splits.append((X_test, test_indices))

    # Update indices and labels for the next iteration
    indices = train_indices
    current_labels = train_labels
k_fold_splits.append((X_train, train_indices))


# If you wish to save each fold to a file:
v_ds_all, t_ds_all = [[] for _ in range(K)], [[] for _ in range(K)]
for i in range(K):
    x_ = k_fold_splits[i][0]
    v_ds_all[i]=x_.copy()
    for j in range(K):
        if i != j:
            t_ds_all[j].extend(x_.copy())

ids_comb = set([comb_['id'] for comb_ in combined_data])
ids_valid = []
for v_ds in v_ds_all: ids_valid.extend([v_['id'] for v_ in v_ds])

for i in range(K):
    ids_valid = [t_['id'] for t_ in t_ds_all[i]];
    ids_valid.extend(v_['id'] for v_ in v_ds_all[i])
    #assert ids_comb == set(ids_valid)

for i in range(K):
    save_jsonl(t_ds_all[i], f"{data_path}/train_fold_{i}.jsonl")
    save_jsonl(v_ds_all[i], f"{data_path}/dev_fold_{i}.jsonl")
