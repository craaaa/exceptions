import argparse
from tqdm import tqdm
from datasets import load_dataset
import pickle
import glob
import os
from pathlib import Path
import spacy
import spacy_transformers
from spacy.matcher import Matcher

parser = argparse.ArgumentParser(description="Split dataset into chunks")
parser.add_argument('--dataset_name', type=str, help='name of a huggingface dataset')
parser.add_argument("--corpus_dir", type=str, help='prefix of path to corpus', default='/scratch/cl5625/openwebtext')
parser.add_argument("--corpus_prefix", type=str, help='split number', default="urlsf_subset06")

args = parser.parse_args()
texts = []

if args.dataset_name:
    dataset_name = args.dataset_name
    dataset = load_dataset(dataset_name)

    for line in tqdm(dataset['train']):
        texts.append(line['text'])
elif args.corpus_prefix:
    dataset_name = 'openwebtext'
    for file in tqdm(glob.glob(args.corpus_dir + "/" + args.corpus_prefix + "*")):
        with open(file, 'r') as f:
            for line in f:
                texts.append(line)

spacy.require_gpu()
nlp = spacy.load("en_core_web_trf")

matcher = Matcher(nlp.vocab)
cost_verbs = [{"LEMMA": {"IN": ["earn", "cost", "fetch"]}}]
take_verbs = [{"LEMMA": {"IN": ["take", "need", "last", "require"]}}]
benefit_verbs = [{"LEMMA": {"IN": ["benefit", "help", "profit", "strengthen"]}}]
ooze_verbs = [{"LEMMA": {"IN": ["discharge", "emanate", "emit", "radiate"]}}]
match_verbs = [{"LEMMA": {"IN": ["approximate", "match", "mirror", "resemble"]}}]

matcher.add("cost_verb", [cost_verbs])
matcher.add("take_verb", [take_verbs])
matcher.add("benefit_verb", [benefit_verbs])
matcher.add("ooze_verb", [ooze_verbs])
matcher.add("match_verb", [match_verbs])

counts = {}
others = []
for doc in tqdm(nlp.pipe(texts, batch_size=50)):
    for sent in doc.sents:
        matches = matcher(sent)
        if not matches:
            continue
        for match_id, start, end in matches:
            for token in sent[start:end]:
                if token.pos_ != "VERB":
                    continue
                if token.lemma_ not in counts.keys():
                    counts[token.lemma_] = {"all": 0, "act": 0, "pass": 0}
                counts[token.lemma_]["all"] += 1
                child_deps = [child.dep_ for child in token.children]
                # if sentence is passive
                if "nsubjpass" in child_deps or "csubjpass" in child_deps or "auxpass" in child_deps:
                    counts[token.lemma_]["pass"] += 1
                # if sentence is active transitive?
                elif "nsubj" in child_deps and "dobj" in child_deps and "dative" not in child_deps:
                    counts[token.lemma_]["act"] += 1
                else:
                    others.append(sent.text)

print(counts)
with open('/scratch/cl5625/corpus-splits/{}/{}_texts_{}.txt'.format(dataset_name, dataset_name, args.corpus_prefix), 'w') as f:
    f.write(str(counts))
    for line in others:
        f.write(line)

print("Saved dataset.")