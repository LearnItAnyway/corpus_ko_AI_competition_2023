for P in '1.3b_r'; do  #'5.8b', '12.8b'
for I in 0 1 2 3 4 ; do 
MM=./polyglot-ko-1.3b_gptq  #
#MM=microsoft/deberta-v2-xxlarge-mnli
deepspeed train_model_cls.py \
	--model_name_or_path $MM \
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
	--metric_for_best_model "eval_loss" --greater_is_better False --problem_type 'regression'
mv trainable_params_${I}.pth ${P}_${I}.pt
rm output -rf
done
done
#done
#done
#--model_name_or_path  ./polyglot-ko-1.3b_gptq \
#--model_name_or_path  EleutherAI/polyglot-ko-1.3b \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
#--data_path ./NIKL_SC_2023/train_${I}.jsonl \
