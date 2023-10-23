import json
import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPTQConfig
from tqdm import trange

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--llm', type=str, default='./output')
parser.add_argument('--data_path', type=str, default='NIKL_EA_2023/nikluge-ea-2023-dev.jsonl')
parser.add_argument('--lora', type=str, default='')
parser.add_argument('--save_path', type=str, default='./results')
parser.add_argument('--save_name', type=str, default='_')
parser.add_argument('--batch_size', type=int, default=32)


args = parser.parse_args()

device='cuda'
llm = args.llm
data_path = args.data_path
save_test_path = f'{args.save_path}/test_{args.save_name}.npy'
save_test_path_ = f'{args.save_path}/test_{args.save_name}_f.npy'
save_dev_path = f'{args.save_path}/dev_{args.save_name}.npy'
save_dev_path_ = f'{args.save_path}/dev_{args.save_name}_f.npy'
batch_size=args.batch_size
#saved_params = torch.load("5.8b_1.pt")
test_path = 'NIKL_EA_2023/nikluge-ea-2023-test.jsonl'

tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=False)
if llm.find('poly')!=-1:
    tokenizer.add_special_tokens({
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<unk>",
    })
else:
#if model_args.model_name_or_path.find('bert') != -1:
    tokenizer.add_special_tokens({
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "[UNK]", "pad_token": "[PAD]",
    })
TA1, TA2 = '<stage_1>', '<stage_2>'

tokenizer.add_tokens([TA1, TA2])

## Load Model
class LoRALinear(nn.Module):
    def __init__(self, linear, rank=64, lora_weight=0.5, ):
        super().__init__()
        self.linear = linear
        try:
            self.lora = nn.Sequential(nn.Linear(linear.in_features, rank, False),
                                    nn.Linear(rank, linear.out_features, False))
        except:
            self.lora = nn.Sequential(nn.Linear(linear.infeatures, rank, False),
                                    nn.Linear(rank, linear.outfeatures, False))
        self.linear.requires_grad_(False)
        self.lora_weight=lora_weight

    def forward(self, x):
        x_linear = self.linear(x)
        x_lora = self.lora(x)
        return x_linear+x_lora*self.lora_weight

if llm.find('gptq') !=-1:
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForSequenceClassification.from_pretrained(
                llm, num_labels=8,
                device_map='auto', quantization_config=gptq_config)
    saved_params = torch.load(args.lora)
    for k in saved_params.keys(): print(saved_params[k].shape, saved_params[k].dtype, saved_params[k].device)
    model.resize_token_embeddings(len(tokenizer.get_vocab().keys())+1)

    for i in range(len(model.gpt_neox.layers)):
        model.gpt_neox.layers[i].attention.query_key_value = LoRALinear(model.gpt_neox.layers[i].attention.query_key_value)
        model.gpt_neox.layers[i].attention.dense = LoRALinear(model.gpt_neox.layers[i].attention.dense)
        model.gpt_neox.layers[i].mlp.dense_h_to_4h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_h_to_4h)
        model.gpt_neox.layers[i].mlp.dense_4h_to_h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_4h_to_h)

    for name, param in model.named_parameters():
        if name in saved_params.keys():
            param.data = saved_params[name]
    model.resize_token_embeddings(len(tokenizer.get_vocab().keys())+1)
    bbb = torch.LongTensor([[0, 1, 2]]).cuda()
else:
    model = AutoModelForSequenceClassification.from_pretrained(
                llm, num_labels=8, device_map=device)

model.config.pad_token_id=tokenizer.pad_token_id
model.config.problem_type = "multi_label_classification"
model.config.use_cache = False

model.eval()

## Load data
lines = []
with open(data_path, 'r') as file:
    for line in file:
        lines.append(json.loads(line))

lines_test = []
with open(test_path, 'r') as file:
    for line in file:
        lines_test.append(json.loads(line))



def preprocess(input, target, tokenizer, label=None):
    s_ = f'{input}<stage_1>{target}<stage_2>'
    input_id = tokenizer.encode(s_, add_special_tokens=False)
    if label is not None:
        label_tensor = []
        for key in ['joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']:
            label_tensor.append(label[key]=='True')
        return dict(input_ids=torch.LongTensor(input_id), labels=torch.FloatTensor(label_tensor))
    else:
        return dict(input_ids=torch.LongTensor(input_id))

def batch_preprocess(data, tokenizer, label_exist=True):
    if label_exist:
        processed_data = [preprocess(d['input']['form'], d['input']['target']['form'], tokenizer, d['output']) for d in data]
    else:
        processed_data = [preprocess(d['input']['form'], d['input']['target']['form'], tokenizer) for d in data]


    # Manually pad the input_ids
    max_length = max([len(item['input_ids']) for item in processed_data])
    input_ids = [torch.cat([item['input_ids'], torch.LongTensor([tokenizer.pad_token_id] * (max_length - len(item['input_ids'])))]) for item in processed_data]
    input_ids = torch.stack(input_ids)

    # If your data contains labels, pad those as well
    labels = None
    if 'labels' in processed_data[0]:
        labels = torch.stack([d['labels'] for d in processed_data])

    return dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(tokenizer.pad_token_id))


aa, bb = [], []
bb_f = []
for i in trange(0, len(lines), batch_size):
    batch_data = lines[i:i+batch_size]

    batch = batch_preprocess(batch_data, tokenizer)
    inputs = batch['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(inputs, inputs.ne(tokenizer.pad_token_id), output_hidden_states=True)
        logits = outputs['logits']
        sigmoid_outputs = torch.sigmoid(logits).detach().cpu().numpy()
        seq_len = torch.eq(inputs, model.config.pad_token_id).long().argmax(-1)-1
        features = outputs['hidden_states'][-1][torch.arange(len(seq_len)).cuda(), seq_len].detach().cpu().numpy()
        #['logits']


    for j, data in enumerate(batch_data):
        label = data['output']
        label_tensor = [label[key]=='True' for key in ['joy', 'anticipation', 'trust', 'surprise', 'disgust', 'fear', 'anger', 'sadness']]

        aa.append(np.array(label_tensor))
        bb.append(sigmoid_outputs[j])
        bb_f.append(features[j])


cc, cc_f = [], []
for i in trange(0, len(lines_test), batch_size):
    batch_data = lines_test[i:i+batch_size]

    batch = batch_preprocess(batch_data, tokenizer, label_exist=False)
    inputs = batch['input_ids'].to(device)
    with torch.no_grad():
        outputs = model(inputs, inputs.ne(tokenizer.pad_token_id), output_hidden_states=True)
        logits = outputs['logits']
        sigmoid_outputs = torch.sigmoid(logits).detach().cpu().numpy()
        seq_len = torch.eq(inputs, model.config.pad_token_id).long().argmax(-1)-1
        features = outputs['hidden_states'][-1][torch.arange(len(seq_len)).cuda(), seq_len].detach().cpu().numpy()

    for j, data in enumerate(batch_data):
        cc.append(sigmoid_outputs[j])
        cc_f.append(features[j])


aa_, bb_, cc_ = np.array(aa), np.array(bb), np.array(cc)


np.save(save_test_path, cc_)
np.save(save_dev_path, np.array([aa_, bb_]))

np.save(save_test_path_, np.array(cc_f))
np.save(save_dev_path_, np.array(bb_f))
