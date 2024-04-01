#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=train
#SBATCH --output=./logs/%j_%x.out
#SBATCH --error=./logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --time=47:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo
singularity exec --nv --overlay /scratch/cl5625/overlay-exceptions.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "

source /ext3/env.sh
conda activate exceptions

export WANDB_PROJECT='exceptions_counterfactuals'
export WANDB_LOG_MODEL='end'
echo dataset: ${DATASET_NAME}
echo seed: $SEED
python scripts/train.py \
    --model_type gpt2 \
    --output_dir /scratch/cl5625/exceptions/models/${DATASET_NAME}_$SEED \
    --train_file /scratch/cl5625/exceptions/data/${DATASET_NAME}/train_100M.txt \
    --validation_file /scratch/cl5625/exceptions/data/${DATASET_NAME}/validation.txt \
    --seed $SEED \
    --report_to wandb \
    --load_best_model_at_end true \
    --tokenizer_name gpt2 \
    --run_name ${DATASET_NAME}_${SEED} \
    --do_train \
    --do_eval \
    --dataset_config_name unshuffled_deduplicated_no \
    --dataloader_pin_memory true \
    --preprocessing_num_workers 16 \
    --fp16 true \
    --block_size 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 5 \
    --learning_rate 6e-4 \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --weight_decay 0.01 \
    --num_train_epochs 40 \
    --logging_steps 50 \
    --save_steps 10000 \
    --save_total_limit 5 \
    --evaluation_strategy steps \
    --eval_steps 1000    \
"