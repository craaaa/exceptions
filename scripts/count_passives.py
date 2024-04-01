import argparse
from corpus_utils import *
from tqdm import tqdm
from datasets import load_dataset
import pickle
import glob
import os
from pathlib import Path
from spacy.matcher import Matcher

parser = argparse.ArgumentParser(description="Split dataset into chunks")
parser.add_argument('--dataset_name', type=str, help='name of a huggingface dataset')
parser.add_argument("--corpus_file", type=str, help='prefix of path to corpus', default='/scratch/cl5625/exceptions/data/100M/train_100M.txt')

args = parser.parse_args()
texts = []

if args.dataset_name:
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name)

    for line in tqdm(dataset['train']):
        texts.append(line['text'])
else:
    dataset_name = 'openwebtext'
    #for file in tqdm(glob.glob(args.corpus_dir + "/" + "*")):
    with open(args.corpus_file, 'r') as f:
        for line in f:
            texts.append(line)

#nlp = spacy.load("en_core_web_trf", enable=["tagger", "morphologizer", "trainable_lemmatizer", "parser"])

nlp = load_model()

take_verbs = ["take", "need", "last", "require"]
agent_patient = ["hit", "wash", "carry", "push", "drop"]
exp_theme = ["see", "hear", "like", "know", "remember"]

matcher = load_matcher(nlp, take_verbs + agent_patient + exp_theme)

counts = {}
sentences = []

with open('/scratch/cl5625/exceptions/data/100M/a_others.txt', 'w') as f:
    for doc in tqdm(nlp.pipe(texts, batch_size=1000), total=len(texts)):
        for sent in doc.sents:
            matches = matcher(sent)
            if not matches:
                continue
            for match_id, start, end in matches:
                for token in sent[start:end]:
                    if token.pos_ != "VERB":
                        continue
                    if token.lemma_ not in counts.keys():
                        counts[token.lemma_] = {"all": 0, "active": 0, "passive": 0, "other": 0}
                    counts[token.lemma_]["all"] += 1
                    child_deps = [child.dep_ for child in token.children]
                    # if sentence is passive
                    if has_active_deps(child_deps):
                        sent_type = 'active'
                    # if sentence is active transitive?
                    elif has_passive_deps(child_deps):
                        sent_type = 'passive'
                    else:
                        sent_type = 'other'
                    counts[token.lemma_][sent_type] += 1
                    line = f"{sent_type}, {token.lemma_}, {sent}\n"
                    f.write(line)

print(counts)
with open('/scratch/cl5625/exceptions/data/100M/a_counts.txt', 'w') as f:
    f.write(str(counts))

print("Saved dataset.")