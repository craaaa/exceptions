import argparse
import csv
import json
from minicons import scorer
import numpy as np
from pathlib import Path
import re
import statistics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_latest_checkpoint(model_dir):
    checkpoint_dirs = list(model_dir.glob('*/'))
    print(checkpoint_dirs)
    pattern = re.compile(r'checkpoint-(\d+)$')
    checkpoint_nos = [int(pattern.search(str(dir)).group(1)) for dir in checkpoint_dirs]
    print(checkpoint_nos)
    largest_checkpoint = max(checkpoint_nos)
    return model_dir / f'checkpoint-{largest_checkpoint}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into chunks")
    parser.add_argument("--model_dir", type=str, help='Model name.', default='/scratch/cl5625/exceptions/models')
    parser.add_argument("--model_name", type=str, help='Model name.', default='last_to_carry_frequency_1421')

    args = parser.parse_args()

    model_name = args.model_name
    model_path = Path(args.model_dir) / model_name
    model_checkpoint = get_latest_checkpoint(model_path)
    #model_checkpoint = 'openai-community/gpt2'
    print(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, return_dict=True)
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    model = scorer.IncrementalLMScorer(model, tokenizer=tokenizer, device='cpu')

    test_data = Path("/scratch/cl5625/exceptions/data/test_sentences.csv")
    stimuli = {}
    with open(test_data, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            verb_class = row['verb_class']
            if verb_class not in stimuli.keys():
                stimuli[verb_class] = []
            stimuli[verb_class].append([row['active'], row['passive'], row['verb'], row['frame_no']])

    results = []
    for verb_class, verb_class_examples in stimuli.items():
        stimuli_dl = DataLoader(verb_class_examples, batch_size = 100)

        for batch in stimuli_dl:
            good, bad, verbs, frames = batch
            good_scores = model.sequence_score(good, reduction = lambda x: x.sum(0))
            bad_scores = model.sequence_score(bad, reduction = lambda x: x.sum(0))
            pass_drop = [g - b for g,b in zip(good_scores, bad_scores)]
            results.extend(zip(good_scores, bad_scores, pass_drop, verbs, frames))

    with open(f"/scratch/cl5625/exceptions/scores/minicons/{model_name}.csv", "w") as o:
        o.write("active_score,passive_score,pass_drop,verb,frame\n")
        for (active_score, passive_score, pass_drop, verb, frame) in results:
            o.write(f"{active_score},{passive_score},{pass_drop},{verb},{frame}\n")



    exit()
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