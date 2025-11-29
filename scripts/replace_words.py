import argparse
from tqdm import tqdm
from datasets import load_dataset
import pickle
import glob
import os
import random
from pathlib import Path
import spacy
import spacy_transformers
from spacy.matcher import Matcher

parser = argparse.ArgumentParser(description="Split dataset into chunks")
parser.add_argument('--dataset_name', type=str, help='name of a huggingface dataset')
parser.add_argument("--corpus_dir", type=str, help='prefix of path to corpus', default='/Users/cl5625/nanoGPT/data/openwebtext_tiny')

args = parser.parse_args()
texts = []

spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")
matcher = Matcher(nlp.vocab)
agent_patient = [{"LEMMA": {"IN": ["hit", "wash", "carry", "push", "drop"]}}]
exp_theme = [{"LEMMA": {"IN": ["see", "hear", "like", "know", "remember"]}}]
# take_verbs = [{"LEMMA": {"IN": ["take", "need", "last", "require"]}}]
# benefit_verbs = [{"LEMMA": {"IN": ["benefit", "help", "profit", "strengthen"]}}]
# ooze_verbs = [{"LEMMA": {"IN": ["discharge", "emanate", "emit", "radiate"]}}]
# match_verbs = [{"LEMMA": {"IN": ["approximate", "match", "mirror", "resemble"]}}]

matcher.add("agent_patient", [agent_patient])
# matcher.add("take_verb", [take_verbs])
# matcher.add("benefit_verb", [benefit_verbs])
# matcher.add("ooze_verb", [ooze_verbs])
# matcher.add("match_verb", [match_verbs])

def has_passive_deps(deps):
    return "nsubjpass" in deps or "csubjpass" in deps or "auxpass" in deps

def has_active_deps(deps):
    return "nsubj" in deps and "dobj" in deps and "dative" not in deps

def flip_coin(prob=0.5):
    random.random() < prob

def get_save_matches(save_filename, texts):
    counts = {}
    with open(save_filename, 'w') as f:
        for doc in tqdm(nlp.pipe(texts, batch_size=50)):
            to_write = ""
            for sent in doc.sents:
                matches = matcher(sent)
                if not matches:
                    to_write += sent.text + " "
                    continue
                verb_tokens = [token for (_, start, end) in matches for token in sent[start:end] if token.pos_ == "VERB"]
                sent_child_deps = []
                for token in verb_tokens:
                    if token.lemma_ not in counts.keys():
                        counts[token.lemma_] = {"all": 0, "act": 0, "pass": 0, "removed": 0}
                    counts[token.lemma_]["all"] += 1
                    child_deps = [child.dep_ for child in token.children]
                    # if sentence is passive
                    if has_passive_deps(child_deps):
                        counts[token.lemma_]["pass"] += 1
                    # if sentence is active transitive?
                    elif has_active_deps(child_deps):
                        counts[token.lemma_]["act"] += 1
                    sent_child_deps += child_deps
                chance = flip_coin(0.5) if has_passive_deps(sent_child_deps) else 1
                if chance:
                    to_write += sent.text + " "
                else:
                    print(sent.text)
                    for token in verb_tokens:
                        counts[token.lemma_]["removed"] += 1
            f.write(to_write.strip() + "\n\n")
                    
    print(f"Saved dataset to {save_filename}.")
    return counts

if args.dataset_name:
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name)

    for line in tqdm(dataset['train']):
        texts.append(line['text'])
    get_save_matches(os.path.join(args.corpus_dir, dataset_name + ".txt"), texts)
elif args.corpus_dir:
    dataset_name = 'openwebtext'
    for filename in tqdm(glob.glob(os.path.join(args.corpus_dir, "*.csv"))):
        base_filename = os.path.basename(filename)
        with open(filename, 'r') as f:
            for line in f:
                texts.append(line)
            get_save_matches(os.path.join(args.corpus_dir, os.path.splitext(base_filename)[0] + "_processed.txt"), texts)