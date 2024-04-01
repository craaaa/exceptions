import argparse
from corpus_utils import *
import json
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path

def get_save_matches(save_filename, removed_filename, texts, max_passive):
    counts = {"all": 0, "act": 0, "pass": 0, "other": 0, "kept": 0, "removed": 0}
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
                    chance = flip_coin(0.2) if counts["kept"] < max_passive else False
                    if chance:
                        to_write += sent.text + " "
                        counts["kept"] += len(verb_tokens)
                    else:
                        removed.append(sent.text)
                        counts["removed"] += len(verb_tokens)
                elif has_active_deps(child_deps):
                    counts["act"] += len(verb_tokens)
                    to_write += sent.text + " "
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
    parser.add_argument("--target_dir", type=str, help='directory to save dataset', default='/scratch/cl5625/exceptions/data/')
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
    target_value = verb_counts[source_verb]["passive"]
    print(f"Removing passives of {target_verb} to reach target value of {source_verb}: {target_value}")

    for filename in corpus_dir.glob("*.txt"):
        texts = []
        base_filename = filename.name
        print(f"Analyzing {base_filename}")
        with open(filename, 'r') as f:
            for line in f:
                texts.append(line.strip())
        save_folder = Path(args.target_dir) / f"{source_verb}_to_{target_verb}_frequency"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_file = save_folder / base_filename
        removed_file = save_folder / (filename.stem + "_removed.txt")
        counts = get_save_matches(save_file, removed_file, texts, target_value)
        with open(save_folder / (filename.stem + "_counts.json"), 'w') as f:
            json.dump(counts,f)
        print("Done.")