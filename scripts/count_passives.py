import argparse
import corpus_utils
from tqdm import tqdm
from datasets import load_dataset
import pickle
import glob
import os
from pathlib import Path
import pyrootutils
from spacy.matcher import Matcher

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

def main(dataset_name):
    # make directory for results if it doesn't exist
    results_dir = PROJECT_ROOT / "scores" / "corpus" / dataset_name
    os.makedirs(results_dir, exist_ok=True)

    # load dataset from huggingface
    dataset = load_dataset(dataset_name)
    texts = dataset['train']['text'][:1000]

    nlp = corpus_utils.load_model()
    matcher = corpus_utils.load_matcher(nlp, corpus_utils.all_verbs)

    counts = {}
    sentences = []

    with open(results_dir / 'counts_examples.txt', 'w') as f:
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
                        if corpus_utils.has_active_deps(child_deps):
                            sent_type = 'active'
                        # if sentence is active transitive?
                        elif corpus_utils.has_passive_deps(child_deps):
                            sent_type = 'passive'
                        else:
                            sent_type = 'other'
                        counts[token.lemma_][sent_type] += 1
                        line = f"{sent_type}, {token.lemma_}, {sent}\n"
                        f.write(line)

    print(counts)


    with open(results_dir / 'counts.txt', 'w') as f:
        f.write(str(counts))

    print("Saved dataset.")

if __name__ == "__main__":
    main("craa/100M")