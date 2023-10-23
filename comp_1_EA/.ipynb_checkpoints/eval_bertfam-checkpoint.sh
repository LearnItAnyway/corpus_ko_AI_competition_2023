#MODELS[skt/ko-gpt-trinity-1.2B-v0.5]='kogpt'
declare -A MODELS


MODELS[tunib/electra-ko-base]='o_tkb'
MODELS[tunib/electra-ko-en-base]='o_tkeb'
MODELS[monologg/koelectra-base-finetuned-nsmc]='o_efn'
MODELS[klue/roberta-base]='o_k_rb'
MODELS[monologg/koelectra-base-v3-discriminator]='o_eb'
MODELS[jaehyeong/koelectra-base-v3-generalized-sentiment-analysis]='o_jk_es'
MODELS[nlp04/korean_sentiment_analysis_dataset3_best]='o_ks_rl'
MODELS[klue/roberta-large]='o_k_rl'
MODELS[nlp04/korean_sentiment_analysis_kcelectra]='o_ks_kc'
MODELS[matthewburke/korean_sentiment]='o_m_ks'


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
## Done


MODELS[beomi/kcbert-large]='b_kcbel'


#MODELS[beomi/KcBERT-v2023]='b_kcbe'
#MODELS[HyeonSang/kobert-sentiment]='h_ks'

for MM in "${!MODELS[@]}"; do
echo "${MODELS[$MM]}"
done

for MM in "${!MODELS[@]}"; do
    P="${MODELS[$MM]}"
    for I in 0 1 2 3 4 ; do 
        rm output -rf
        deepspeed train_model_cls.py \
        	--model_name_or_path $MM \
            --data_path ./NIKL_EA_2023/train_fold_${I}.jsonl \
            --output_dir ./output \
            --num_train_epochs 5 \
            --model_max_length 512 \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 16 \
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
            --gradient_checkpointing False \
            --gradient_accumulation_steps 1 \
        	--metric_for_best_model "eval_loss" --greater_is_better False  \
        	--fp16 False  --deepspeed ./ds_config.json  
        	#--fp16 True --half_precision_backend 'cuda_amp'  --deepspeed ./ds_config.json  
        
        CUDA_VISIBLE_DEVICES=0 python eval_1.py --llm ./output --data_path NIKL_EA_2023/dev_fold_${I}.jsonl --save_name ${P}_fold_${I} --batch_size 64
    done
done
#   	for fold in 4; do
#   	#for fold in 0 1 2 3 4; do
#   	python eval_1.py --llm ./polyglot-ko-1.3b_gptq --data_path NIKL_EA_2023/dev_fold_${fold}.jsonl --lora "1.3b_${fold}.pt" --save_name 1.3b_fold_${fold} 
#   	done
