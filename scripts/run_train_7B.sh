export WANDB_MODE="disabled"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port=20002 src/train_lora.py \
    --model_name_or_path llama2_7b_ckpt_dir/ \
    --data_path data/processed/train.json \
    --bf16 False \
    --output_dir outputs/ \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 10 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed configs/stage2.json