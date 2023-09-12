# Language Models Can Learn Exceptions to Syntactic Rules (Leong and Linzen 2023)
This repository contains code for the paper [Language Models Can Learn Exceptions to Syntactic Rules](https://doi.org/10.7275/h25z-0y75).

## Repository Contents
- 100M words training dataset based on [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) (Gokaslan and Cohen 2019): `data/100M`
- Final checkpoints of 5 models trained with early stopping for 30 epochs: `models/100_[n]`
- Training script: `train.py`
- Test sentences: `data/test_sentences.csv`
- Evaluation script: `eval.py`
- Sentence scores: `scores`

## Instructions to replicate

### Setup

```
conda env create -f environment.yml
conda activate traingpt
```

### Training
To train your own model on the 100M word dataset, use the following code: 

```
python train.py \
    --output_dir models/ \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file data/100M/train_100M.txt \
    --validation_file data/100M/validation.txt \
    --dataset_config_name unshuffled_deduplicated_no \
    --bf16 true \
    --do_train true \
    --do_eval true \
    --block_size 512 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 5 \
    --learning_rate 6e-4 \
    --warmup_steps 100 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --weight_decay 0.01 \
    --num_train_epochs 30 \
    --logging_steps 50 \
    --save_steps 1200 \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --load_best_model_at_end true
```

### Evaluation

Run `eval.py` on a test dataset. The script looks into the `models` directory for a model with the given name
and checkpoint and saves sentence scores in `scores`.

```
python eval.py data/test_sentences.csv --model 100M_4 --checkpoint 168000
```