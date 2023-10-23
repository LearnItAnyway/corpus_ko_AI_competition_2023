# Overview
This is solution of [2023년 국립국어원 인공 지능 언어 능력 평가: 이야기 완성 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=102&clCd=END_TASK&subMenuId=sub01).

# Requirements and Model Use
The solutions should be run in the GPU with 24GB, and the pretrained model should be publhsed until Aug, 11. 
So the possible pretrained model is summarized as

- Model having smaller parameter than 9~10B parameters (With half precision)
  - e.g., electra-ko, roberta... 
- gptq model or gptq with lora having smaller parameter than 30B
  - e.g., kullm, polyglot-ko (up to 12.8b), ko-llama-2-7b

We tested polyglot-ko-12.8b, kullm-12.8b, and polyglot-ko-1.3b.
Since the relationship between bce loss and ROUGE score is not clear, we test the model without validation, which may be the reason for the low scores.
(When we splits the train/dev dataset and evaluate the loss, the validation loss decreases when epoch>1. 
However, the leaderboard scores seems increases with the epoch.)

# Solution
## Stage 1 - Model Training

The pretraned model has been fine-tuned as

```
MODEL=./polyglot-ko-1.3b_gptq  # Path of your model 
P='1.3b' # name your model
for ep in 1; do 
    deepspeed train_model.py \
        --model_name_or_path $MM \
        --data_path ./NIKL_SC_2023/comb_train.jsonl \
        --output_dir ./output \
        --num_train_epochs ${ep} \
        --model_max_length 256 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_steps 1 --save_strategy 'epoch'\
        --eval_steps 1 --evaluation_strategy 'epoch' \
        --load_best_model_at_end False \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --warmup_steps 2 \
        --logging_steps 1 \
        --lr_scheduler_type "cosine" \
        --gradient_checkpointing True \
        --deepspeed ./ds_config.json  \
        --gradient_accumulation_steps 1 --fp16 True --half_precision_backend 'cuda_amp' \
        --metric_for_best_model "eval_loss" --greater_is_better False
    mv trainable_params.pth ${P}_${ep}_gptq.pt
done
```
You can change the training options in `train_model_cls.py` and `ds_config.json`

## Stage 2 - Generate Output

Generate output. You can change the number of batch and number of beam which may affect the score

```
python eval_batch.py
```
