#for P in '1.3b'; do  #'5.8b', '12.8b'
#for I in 1; do 
MM=beomi/kcbert-base  #
MM=./polyglot-ko-1.3b_gptq  #
MM="team-lucid/deberta-v3-base-korean"
MM=microsoft/deberta-base-mnli
MM=klue/roberta-large
MM=beomi/kcbert-base  #
MM=klue/roberta-base
#MM=microsoft/deberta-v2-xxlarge-mnli
deepspeed train_model_cls.py \
	--model_name_or_path $MM \
    --data_path ./NIKL_EA_2023/nikluge-ea-2023-train.jsonl \
    --output_dir ./output_klue \
    --num_train_epochs 10 \
    --model_max_length 512 \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 --save_strategy 'steps'\
	--eval_steps 200 --evaluation_strategy 'steps' \
	--load_best_model_at_end True \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --deepspeed ./ds_config.json  \
    --gradient_accumulation_steps 1 --fp16 True --half_precision_backend 'cuda_amp' \
	--metric_for_best_model "eval_f1" --greater_is_better True
#mv trainable_params.pth ${P}_${I}.pt
#rm output -rf
#done
#done
#--model_name_or_path  ./polyglot-ko-1.3b_gptq \
#--model_name_or_path  EleutherAI/polyglot-ko-1.3b \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
#--data_path ./NIKL_SC_2023/train_${I}.jsonl \
