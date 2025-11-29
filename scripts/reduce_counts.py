import argparse
from corpus_utils import *
import json
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
import math

def get_save_matches(save_filename, removed_filename, texts, target_active, target_passive):
    counts = {"all": 0, "act": 0, "pass": 0, "other": 0, "kept_act": 0, "kept_pass": 0, "removed_act": 0, "removed_pass": 0}
    removed = []
    with open(save_filename, 'w') as f:
        for doc in tqdm(nlp.pipe(texts, batch_size=1000), total=len(texts)):
            to_write = ""
            for sent in doc.sents:
                matches = matcher(sent)
                if not matches:
                    to_write += sent.text + " "
                    continue
                verb_tokens = [token for (_, start, end) in matches for token in sent[start:end] if token.pos_ == "VERB"]
                counts["all"] += len(verb_tokens)
                child_deps = [child.dep_ for token in verb_tokens for child in token.children ]
                # if sentence is passive
                # if sentence is active transitive?
                if has_passive_deps(child_deps):
                    counts["pass"] += len(verb_tokens)
                    if counts["kept_pass"] < target_passive:
                        to_write += sent.text + " "
                        counts["kept_pass"] += len(verb_tokens)
                    else:
                        removed.append(sent.text)
                        counts["removed_pass"] += len(verb_tokens)
                        print("Removed passive sentence: ", sent.text)
                elif has_active_deps(child_deps):
                    counts["act"] += len(verb_tokens)
                    if counts["kept_act"] < target_active:
                        to_write += sent.text + " "
                        counts["kept_act"] += len(verb_tokens)
                    else:
                        removed.append(sent.text)
                        counts["removed_act"] += len(verb_tokens)
                        print("Removed active sentence: ", sent.text)
                else:
                    counts["other"] += len(verb_tokens)
                    to_write += sent.text + " "
            f.write(to_write.strip() + "\n\n")
                    
    print(f"Saved dataset to {save_filename}.")
    with open(removed_filename, 'w') as f:
        for line in removed:
            f.write(line)
            f.write("\n")
    
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into chunks")
    parser.add_argument("--corpus_dir", type=str, help='prefix of path to corpus', default='/scratch/cl5625/exceptions/data/100M')
    parser.add_argument("--target_dir", type=str, help='directory to save dataset', default='/scratch/cl5625/exceptions/data')
    parser.add_argument("-s", "--source_verb", type=str, help='verb whose frequency to emulate', default='last')
    parser.add_argument("-t", "--target_verb", type=str, help='verb to remove from dataset', default='hit')


    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir)
    source_verb = args.source_verb
    target_verb = args.target_verb

    nlp = load_model(require_gpu=True)
    matcher = load_matcher(nlp, [target_verb])

    with open(corpus_dir / "counts.json", 'r') as f:
        verb_counts = json.load(f)
    original_active = verb_counts[target_verb]["active"]
    original_passive = verb_counts[target_verb]["passive"]
    original_ratio = original_passive / original_active

    source_active = verb_counts[source_verb]["active"]
    source_passive = verb_counts[source_verb]["passive"]
    source_ratio = source_passive / source_active
    
    print(f"Removing passives of {target_verb} [active: {original_active}, passive: {original_passive}, ratio: {original_ratio}]")
    print(f"to reach target ratio of {source_verb} [active: {source_active}, passive: {source_passive}, ratio: {source_ratio}]")

    # calculate target number of actives and passives
    # number of actives needed if passives are all kept
    target_passive = original_passive
    target_active = int(math.floor(original_passive / source_ratio))

    # if too many actives are needed to keep ratio, remove some passives
    if target_active > original_active:
        target_passive = int(math.floor(original_active * source_ratio))
        target_active = int(math.floor(target_passive / source_ratio))

    print(f"Goal: {target_active} actives and {target_passive} passives of {target_verb}; ratio: {target_passive / target_active}")

    assert target_active <= original_active
    assert target_passive <= original_passive


    for filename in [corpus_dir / "train_100M.txt", corpus_dir / "validation.txt"]:
        texts = []
        base_filename = filename.name
        print(f"Analyzing {base_filename}")
        with open(filename, 'r') as f:
            for line in f:
                texts.append(line.strip())
        save_folder = Path(args.target_dir) / f"{source_verb}_to_{target_verb}_frequency"
        save_folder.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {save_folder}")
        save_file = save_folder / base_filename
        removed_file = save_folder / (filename.stem + "_removed.txt")
        counts = get_save_matches(save_file, removed_file, texts, target_active, target_passive)
        with open(save_folder / (filename.stem + "_counts.json"), 'w') as f:
            json.dump(counts,f)
        print("Done.")