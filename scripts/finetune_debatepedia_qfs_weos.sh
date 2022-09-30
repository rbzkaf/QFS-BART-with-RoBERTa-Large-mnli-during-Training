# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# you need to specify data_dir, output_dir and model_name_or_path
export DATA_DIR='data'
export OUTPUT_DIR='trained_model_concat'

python -u train_qfs.py \
    --data_dir $DATA_DIR \
    --baseline \
    --raw \
    --output_dir $OUTPUT_DIR \
    --model_name_or_path 'facebook/bart-large-xsum' \
    --learning_rate 3e-5 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.5 \
    --max_source_length 142 \
    --max_target_length 48 \
    --val_max_target_length 48 \
    --test_max_target_length 48 \
    --freeze_embeds \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --early_stopping_patience 20 \

