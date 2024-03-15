import json
from minicons import scorer
import numpy as np
from pathlib import Path
import statistics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = '100M_5'
model_checkpoint = f'models/{model_name}/checkpoint-166800'
model = AutoModelForCausalLM.from_pretrained(model_checkpoint, return_dict=True)
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device='cuda')

blimp_directory = Path("/scratch/cl5625/blimp/data")
accuracies = []
for test_path in tqdm(blimp_directory.iterdir(),total=100):
    test = test_path.stem
    stimuli = []
    with open(test_path, "r") as f:
        for line in f:
            row = json.loads(line)
            stimuli.append([row['sentence_good'], row['sentence_bad']])
            # stimuli.append([row['sentence_good'], row['sentence_bad']])

    stimuli_dl = DataLoader(stimuli, batch_size = 100)

    results = []
    for batch in stimuli_dl:
        good, bad = batch
        good_scores = model.sequence_score(good, reduction = lambda x: x.sum(0))
        bad_scores = model.sequence_score(bad, reduction = lambda x: x.sum(0))
        results.extend(zip(good_scores, bad_scores))

    def accuracy(data):
        return np.mean([g > b for g,b in data])

    # Computing accuracy of our model:
    acc = accuracy(results)
    # print(f"The accuracy of the model on {test} is {round(acc*100, 2)}%")
    accuracies.append((test, acc))

with open(f"scores/blimp/{model_name}.csv", "w") as o:
    o.write("test,accuracy")
    for test, acc in accuracies:
        o.write(f"{test},{acc}\n")