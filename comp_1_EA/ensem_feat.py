import json
import torch
import copy
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPTQConfig
from sklearn.model_selection import train_test_split
from tqdm import trange
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--hdim', type=int, default=256)
args = parser.parse_args()

th = 0.5
fold = args.fold
k_folds = args.K

aa_, bb_= np.load(f'results/dev_1.3b_fold_{fold}.npy')

models = ['your model 1', 'your model 2' ... ]
""" our models are
Note that there are redundent models that may not improve the actual performance.
But we just use all the trained model as some sort of recycle.
models = ['1.3b', '5.8b', '12.8b', # gptq lora with rank=64, es with validation f1_score
        '5.8b_n', '12.8b_n', # gptq lora with rank=64, es with validation bce loss
        'f_1.3b', # full finetuning, es with validation bce loss
        'f_5.8b', 'f_12.8b', # gptq lora with rank=512, es with validation bce loss,
                             # no significant improvement compared with rank=64
        'f_k5.8b', 'f_k12.8b', # gptq lora with rank=512, es with validation bce loss,
                               # slightly lower performance compared with  polyglot-ko
        'f_kl7b', # gptq lora with rank=512, lower compared with polyglot-ko 5.8b, es with bce loss
        #The following is the models that we fine tune without lora
       'tkb', 'tkeb', 'efn', 'k_rb', 'eb', 'jk_es', 'ks_rl', 'k_rl', 'ks_kc', 'm_ks',]
        #MODELS[tunib/electra-ko-base]='tkb'
        #MODELS[tunib/electra-ko-en-base]='tkeb'
        #MODELS[monologg/koelectra-base-finetuned-nsmc]='efn'
        #MODELS[klue/roberta-base]='k_rb'
        #MODELS[monologg/koelectra-base-v3-discriminator]='eb'
        #MODELS[jaehyeong/koelectra-base-v3-generalized-sentiment-analysis]='jk_es'
        #MODELS[nlp04/korean_sentiment_analysis_dataset3_best]='ks_rl'
        #MODELS[klue/roberta-large]='k_rl'
        #MODELS[nlp04/korean_sentiment_analysis_kcelectra]='ks_kc'
        #MODELS[matthewburke/korean_sentiment]='m_ks'
       # o indicates it is trained with mse loss
       'o_eb', 'o_efn', 'o_jk_es', 'o_k_rb', 'o_k_rl', 'o_ks_kc', 'o_ks_rl', 'o_m_ks',  'o_tkb', 'o_tkeb',
"""

ff_ = [np.load(f'results/dev_{m}_fold_{fold}_f.npy') for m in models]
len_all = [f.shape[-1] for f in ff_]
ff_ = np.hstack(ff_)
ff_t = np.hstack([np.load(f'results/test_{m}_fold_{fold}_f.npy') for m in models])

def micro_f1(y_pred, y_true, epsilon=1e-7):
    tp = torch.sum(y_true * y_pred)
    fp = torch.sum((1 - y_true) * y_pred)
    fn = torch.sum(y_true * (1 - y_pred))

    p = tp / (tp + fp + epsilon)
    r = tp / (tp + fn + epsilon)

    f1 = 2 * p * r / (p + r + epsilon)
    return f1


class SimpleNN(nn.Module):
    def __init__(self, h_dim=args.hdim):
        super(SimpleNN, self).__init__()
        #h_dims = [128 for i in range(8)]
        self.layer = nn.Sequential(
            nn.Linear(ff_.shape[-1], h_dim),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(h_dim, h_dim),
            nn.SELU(),
            nn.Dropout(0.2),
            nn.Linear(h_dim, 8),
            )

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

ee = 1/k_folds
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

all_predictions = []
best__ = []
best_xy = [[] for _ in range(k_folds)]

X_all = torch.FloatTensor(ff_).cuda()
y_all = torch.FloatTensor(aa_).cuda()
X_test = torch.FloatTensor(ff_t).cuda()
for fold, (train_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X_all[train_idx], X_all[val_idx]
    y_train, y_val = y_all[train_idx], y_all[val_idx]

    # Model, criterion and optimizer
    model = SimpleNN().cuda()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    best_ = [0, None]

    for epoch in range(2000):
        model.train()
        outputs = model(X_train)
        optimizer.zero_grad()
        #loss = micro_f1_loss(outputs, y_train)
        loss = criterion(outputs, y_train.detach())
        loss.backward()
        optimizer.step()

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                f1_ = micro_f1((outputs > th) * 1.0, y_train)
                outputs_val = model(X_val) > th
                accuracy = torch.mean((outputs_val == y_val).type(torch.FloatTensor))
                f1 = f1_score(y_true=y_val.detach().cpu().numpy(), y_pred = (outputs_val).detach().cpu(), average='micro')
                #f1 = micro_f1(outputs_val * 1.0, y_val)
                w_s = f1.item() * ee + f1.item() * (1 - ee)

                if w_s > best_[0]:
                    best_[0] = w_s
                    best_xy[fold] = [y_val.detach().cpu().numpy(), model(X_val).detach().cpu().numpy()]
                    best_[1] = copy.deepcopy(model.state_dict())
                if f1_ > 0.997: break
            if epoch%10==0:
                print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy.item()}, F1_train: {f1_.item()}, F1_val: {f1.item()}")
    print(fold, best_[0])
    model.load_state_dict(best_[1])
    ## check the best threshold
    best__.append(best_[0])
    all_predictions.append(model(X_test).detach().cpu())

    del best_
    del model, optimizer
    del X_train, y_train

    gc.collect()
    torch.cuda.empty_cache()
print('fin', np.mean(best__))

y_ensemble = torch.mean(torch.stack(all_predictions), dim=0)

np.save(f'results_feat_ens_{args.fold}_{args.hdim}.npy', y_ensemble.detach().cpu().numpy())
aa = np.vstack([b[0] for b in best_xy])
bb = np.vstack([b[1] for b in best_xy])
np.save(f'ab_{args.fold}_{args.hdim}.npy', np.array([aa, bb]))
for th in [0.4, 0.45, 0.5, 0.55]:
    print(th, f1_score(aa, bb>th, average='micro'))

