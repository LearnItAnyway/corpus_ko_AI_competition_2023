# Overview
This repository is solutions of [2023년 국립국어원 인공 지능 언어 능력 평가: 감정 분석 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103&clCd=END_TASK&subMenuId=sub01).

# Requirements and Model Use
The solutions should be run in the GPU with 24GB, and the pretrained model should be publhsed until Aug, 11. 
So the possible pretrained model is summarized as

- Model having smaller parameter than 9~10B parameters (With half precision)
  - e.g., electra-ko, roberta... 
- gptq model or gptq with lora having smaller parameter than 30B
  - e.g., kullm, polyglot-ko (up to 12.8b), ko-llama-2-7b
We use muliple models that satifies these criteria.
Note that different setup for the pretrained model and lora option can be used to boost the performance.

# Solution
Fine-tune multiple model + feature ensembles

## Stage 1 - Model Training

We first split train/dev(validation) sets (5 folds)

```
python split.py
```

Then, the pretraned model has been fine-tuned as

```
MODEL=./polyglot-ko-1.3b_gptq  # Path of your model 
P='1.3b' # name your model
for I in 0 1 2 3 4 ; do  # Train different folds
    deepspeed train_model_cls.py \
    	--model_name_or_path $MODEL \
        --data_path ./NIKL_EA_2023/train_fold_${I}.jsonl \
        --output_dir ./output \
        --num_train_epochs 4 \
        --model_max_length 512 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 --save_strategy 'steps'\
    	--eval_steps 500 --evaluation_strategy 'steps' \
    	--load_best_model_at_end True \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --warmup_steps 2 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --deepspeed ./ds_config.json  \
        --gradient_accumulation_steps 1 --fp16 True --half_precision_backend 'cuda_amp' \
    	--metric_for_best_model "eval_loss" --greater_is_better False
    mv trainable_params_${I}.pth ${P}_${I}.pt
done
```
You can change the training options in `train_model_cls.py` and `ds_config.json`

## Stage 2 - Extract Features

After saving the trainable parameters, we extract the features of data in dev and test files using `eval.py` 

If the model has been trained without lora, you can evaluate using

```
python eval.py --llm ./output --data_path --data_path NIKL_EA_2023/dev_fold_${I}.jsonl  --save_name ${P}_fold_${I} --batch_size 64 
```

If the model has been trained with lora, you can evaluate using

```
python eval.py --llm $MODEL --data_path --data_path NIKL_EA_2023/dev_fold_${I}.jsonl --lora ${P}_${I}.pt --save_name ${P}_fold_${I} --batch_size 64
```

In this stage, features (not logits) has been saved in `results` folder

## Stage 3 - Ensemble

Based on features from multiple models, logits has been trained

```
for H in 64 96 256; do 
    for I in 0 1 2 3 4; do python ensem_feat.py --fold $I --hdim $H; done
done
```

Since the ensembled logits can be overfitted, we use different number of hidden nodes (e.g., 64, 96, 256... ) and average the output.

```
python submit.py
```
