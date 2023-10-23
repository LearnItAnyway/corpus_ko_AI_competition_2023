import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence

import torch

import transformers
from transformers import Trainer
from torch.utils.data import Dataset

#from llava.train.llava_trainer import LLaVATrainer
#from llava.model import *

from PIL import Image
import torch.nn as nn

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<unk>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_image_conv_front: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(seq_1, seq_3, seq_2, tokenizer):
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    #TA1, TA2 = '<stage_1>', '<stage_2>'
    s_ = f'{seq_1}<stage_1>{seq_3}<stage_2>'
    t_ = seq_2
    #s_, t_ = sources[0][0]['value'], sources[0][1]['value']
    input_id = tokenizer.encode(s_, add_special_tokens=False)
    target = [-100 for _ in range(len(input_id))]
    tt = tokenizer.encode(t_, add_special_tokens=False)
    tt.append(tokenizer.eos_token_id)
    target.extend(tt)
    input_id.extend(tt)
    target=target[-2048:]
    input_id=input_id[-2048:]
    return dict(input_ids=torch.LongTensor(input_id), labels=torch.LongTensor(target))


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 ):
        super(LazySupervisedDataset, self).__init__()

        logging.warning("Loading data...")
        if data_path.endswith('.json'):
            list_data_dict = json.load(open(data_path, "r"))#[:100]
        elif data_path.endswith('.jsonl'):
            list_data_dict = []
            with open(data_path, 'r') as file:
                for line in file:
                    list_data_dict.append(json.loads(line))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
       #if isinstance(i, int):
       #    sources = [sources]
       #assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = preprocess(
            sources['input']['sentence1'], sources['input']['sentence3'], sources['output'],
            self.tokenizer)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    if False:
        train_dataset = dataset_cls(tokenizer=tokenizer,
                                    data_path=data_args.data_path,)
        return dict(train_dataset=train_dataset,
                    eval_dataset=None,
                    data_collator=data_collator)
    else:
        train_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path,)
        eval_dataset = dataset_cls(tokenizer=tokenizer, data_path=data_args.data_path.replace('train', 'dev'))

        return dict(train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if model_args.model_name_or_path.find('gptq') != -1:
        gptq_config = transformers.GPTQConfig(bits=4, disable_exllama=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    device_map='cuda',
                    quantization_config = gptq_config)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
        #model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir, device_map='cuda', torch_dtype=torch.float16
            )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.add_special_tokens({
            "bos_token": "<s>", "eos_token": "</s>",
            "unk_token": "<unk>", "pad_token": "<unk>",
        })
    TA1, TA2 = '<stage_1>', '<stage_2>'
    tokenizer.add_tokens([TA1, TA2])
    class LoRALinear(nn.Module):
        def __init__(self, linear, rank=128, lora_weight=0.5, ):
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


    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    model.resize_token_embeddings(len(tokenizer.get_vocab().keys()))

    if model.config.architectures == ['GPTNeoXForCausalLM']:
        for i in range(len(model.gpt_neox.layers)):
            model.gpt_neox.layers[i].attention.query_key_value = LoRALinear(model.gpt_neox.layers[i].attention.query_key_value)
            model.gpt_neox.layers[i].attention.dense = LoRALinear(model.gpt_neox.layers[i].attention.dense)
            model.gpt_neox.layers[i].mlp.dense_h_to_4h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_h_to_4h)
            model.gpt_neox.layers[i].mlp.dense_4h_to_h = LoRALinear(model.gpt_neox.layers[i].mlp.dense_4h_to_h)

    elif True:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn.q_proj = LoRALinear(model.model.layers[i].self_attn.q_proj)
            model.model.layers[i].self_attn.k_proj = LoRALinear(model.model.layers[i].self_attn.k_proj)
            model.model.layers[i].self_attn.v_proj = LoRALinear(model.model.layers[i].self_attn.v_proj)
            model.model.layers[i].self_attn.o_proj = LoRALinear(model.model.layers[i].self_attn.o_proj)

            model.model.layers[i].mlp.gate_proj = LoRALinear(model.model.layers[i].mlp.gate_proj)
            model.model.layers[i].mlp.down_proj = LoRALinear(model.model.layers[i].mlp.down_proj)
            model.model.layers[i].mlp.up_proj = LoRALinear(model.model.layers[i].mlp.up_proj)
    # Loading
   #saved_params = torch.load("trainable_params.pth")
   #for name, param in model.named_parameters():
   #    if name in saved_params.keys():
   #        param.data = saved_params[name]
    saved_params = torch.load("12.8b_1.pt")
    for name, param in model.named_parameters():
        if name in saved_params.keys():
            param.data = saved_params[name]

    for n,p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
        p = p.float()
    
    trainer = Trainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    #callbacks = [transformers.EarlyStoppingCallback(early_stopping_patience=2)],
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if False:
        if model.config.architectures == ['GPTNeoXForCausalLM']:
            for i in range(len(model.gpt_neox.layers)):
                model.gpt_neox.layers[i].attention.query_key_value = model.gpt_neox.layers[i].attention.query_key_value.merge()
                model.gpt_neox.layers[i].attention.dense = model.gpt_neox.layers[i].attention.dense.merge()
                model.gpt_neox.layers[i].mlp.dense_h_to_4h = model.gpt_neox.layers[i].mlp.dense_h_to_4h.merge()
                model.gpt_neox.layers[i].mlp.dense_4h_to_h = model.gpt_neox.layers[i].mlp.dense_4h_to_h.merge()

        elif True:
            for i in range(len(model.model.layers)):
                model.model.layers[i].self_attn.q_proj = model.model.layers[i].self_attn.q_proj.merge()
                model.model.layers[i].self_attn.k_proj = model.model.layers[i].self_attn.k_proj.merge()
                model.model.layers[i].self_attn.v_proj = model.model.layers[i].self_attn.v_proj.merge()
                model.model.layers[i].self_attn.o_proj = model.model.layers[i].self_attn.o_proj.merge()

                model.model.layers[i].mlp.gate_proj = model.model.layers[i].mlp.gate_proj.merge()
                model.model.layers[i].mlp.down_proj = model.model.layers[i].mlp.down_proj.merge()
                model.model.layers[i].mlp.up_proj   = model.model.layers[i].mlp.up_proj.merge()
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)
    else:
        trainable_params = {name: param.data for name, param in model.named_parameters() if param.requires_grad}
        torch.save(trainable_params, "trainable_params.pth")

if __name__ == "__main__":
    train()

