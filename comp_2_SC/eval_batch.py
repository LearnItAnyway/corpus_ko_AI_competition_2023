import json
import torch
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, BartForConditionalGeneration
from tqdm import trange

device='cuda'
llm = './polyglot-ko-12.8b_gptq'


tokenizer = AutoTokenizer.from_pretrained(llm, use_fast=False)
tokenizer.add_special_tokens({
        "bos_token": "<s>", "eos_token": "</s>",
        "unk_token": "<unk>", "pad_token": "<unk>",
    })
TA1, TA2 = '<stage_1>', '<stage_2>'
tokenizer.add_tokens([TA1, TA2])
#
if llm.find('gptq') != -1:
    gptq_config = GPTQConfig(bits=4, disable_exllama=True)
    model = AutoModelForCausalLM.from_pretrained(
            llm,
            device_map="cuda", quantization_config = gptq_config)
else:
    model = AutoModelForCausalLM.from_pretrained(
            llm,
            device_map="cuda", torch_dtype=torch.float16)


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

    def merge(self):
        self.linear.weight.data += self.lora_weight*(self.lora[1].weight@self.lora[0].weight)
        return self.linear

model.resize_token_embeddings(len(tokenizer.get_vocab().keys()))
if True:
    for i in range(len(model.gpt_neox.layers)):
        model.gpt_neox.layers[i].attention.query_key_value = LoRALinear(model.gpt_neox.layers[i].attention.query_key_value)
        model.gpt_neox.layers[i].attention.dense = LoRALinear(model.gpt_neox.layers[i].attention.dense)
        model.gpt_neox.layers[i].mlp.dense_h_to_4h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_h_to_4h)
        model.gpt_neox.layers[i].mlp.dense_4h_to_h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_4h_to_h)
    
    saved_params = torch.load("12.8b_1.pt")
    for name, param in model.named_parameters():
        if name in saved_params.keys():
            param.data = saved_params[name]
else:
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn.q_proj = LoRALinear(model.model.layers[i].self_attn.q_proj)
        model.model.layers[i].self_attn.k_proj = LoRALinear(model.model.layers[i].self_attn.k_proj)
        model.model.layers[i].self_attn.v_proj = LoRALinear(model.model.layers[i].self_attn.v_proj)
        model.model.layers[i].self_attn.o_proj = LoRALinear(model.model.layers[i].self_attn.o_proj)

        model.model.layers[i].mlp.gate_proj = LoRALinear(model.model.layers[i].mlp.gate_proj)
        model.model.layers[i].mlp.down_proj = LoRALinear(model.model.layers[i].mlp.down_proj)
        model.model.layers[i].mlp.up_proj = LoRALinear(model.model.layers[i].mlp.up_proj)
    saved_params = torch.load("13b_1.pt")
    for name, param in model.named_parameters():
        if name in saved_params.keys():
            param.data = saved_params[name]

data_path = 'NIKL_SC_2023/nikluge-sc-2023-test.jsonl'
lines = []
with open(data_path, 'r') as file:
    for line in file:
        lines.append(json.loads(line))

model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

import pandas as pd
texts = []
for i in trange(len(lines)):
    ii = lines[i]['input']
    sen1, sen3 = ii['sentence1'], ii['sentence3']
    s_ = f"<s>{sen1}<stage_1>{sen3}<stage_2>"
    texts.append(s_)

bbb=2
nb =16
def generate_responses(texts, batch_size=bbb):
    """
    Generate responses for a list of texts using GPT-2, batched by similar token lengths.
    
    Parameters:
        texts (list): A list of input texts.
        batch_size (int): Number of texts to process in a single batch.
    
    Returns:
        List of generated texts corresponding to each input.
    """
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.pad_token  # to avoid an error

    # Compute token lengths and sort texts by it
    token_lengths = [len(tokenizer.encode(text)) for text in texts]
    sorted_texts_with_indices = sorted(enumerate(texts), key=lambda x: token_lengths[x[0]])
    sorted_texts = [item[1] for item in sorted_texts_with_indices]
    original_order_indices = [item[0] for item in sorted_texts_with_indices]

    generated_responses_sorted = []
    for i in trange(0, len(sorted_texts), batch_size):
        batch_texts = sorted_texts[i:i + batch_size]
        encoding = tokenizer(batch_texts, return_tensors='pt', padding=True).to(device)
        try:
            encoding.pop('token_type_ids')
        except:
            pass
            
        #import ipdb;ipdb.set_trace()
        with torch.no_grad():
            generated_ids = model.generate(**encoding, max_new_tokens=bbb*32, num_beams=nb,  do_sample=True)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        generated_responses_sorted.extend([text.split('<stage_2>')[-1] for text in generated_texts])
        print([text.split('<stage_2>')[-1] for text in generated_texts])

    # Rearrange generated responses to the original order
    generated_responses = [text for _, text in sorted(zip(original_order_indices, generated_responses_sorted))]

    return generated_responses

outputs = generate_responses(texts)

# Check a few outputs:
for inp, out in zip(texts[:5], outputs[:5]):
    print(f"Input: {inp}\nOutput: {out}\n{'-'*50}")

#import ipdb;ipdb.set_trace()
for i, o in enumerate(outputs):
    if o[0]==' ': o=o[1:]
    lines[i]['output'] = o

with open('data_gptq.jsonl', 'w', encoding='utf-8') as file:
    for entry in lines:
        file.write(json.dumps(entry, ensure_ascii=False) + '\n')
#        inputs = tokenizer.encode(s_, add_special_tokens=False, return_tensors='pt').to(device)[:, -1024:]
#        with torch.no_grad():
#            output = model.generate(inputs, max_new_tokens=64, num_beams=1)
#        text = tokenizer.batch_decode(output, skip_special_tokens=True)
#        output_text = text[0].split('<stage_2>')[-1]
#        aa.append(output_text)
#        bb.append(lines[i]['output'])
#        df = pd.DataFrame.from_dict({'output': aa, 'label':bb})
#        df.to_csv('tmp.csv')
#    
#        print(lines[i]['output'], output_text)
#    
#    import ipdb; ipdb.set_trace()  # ddongsun
