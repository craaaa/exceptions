import argparse
from corpus_utils import *
import json
from tqdm import tqdm
import random
from datasets import load_dataset
from pathlib import Path


def get_save_matches(save_filename, matcher, target_word, removed_filename, texts):
    counts = {"all": 0, "act": 0, "pass": 0, "other": 0, "kept": 0, "changed": 0}
    removed = []
    with open(save_filename, 'w') as f:
        for doc in tqdm(nlp.pipe(texts, batch_size=1000), total=len(texts)):
            to_write = ""
            for sent in doc.sents:
                matches = matcher(sent)
                if not matches:
                    to_write += sent.text + " "
                    continue
                verb_tokens = [(token, start, end) for (_, start, end) in matches for token in sent[start:end] if token.pos_ == "VERB"]
                counts["all"] += len(verb_tokens)
                child_deps = [child.dep_ for (token,_,_) in verb_tokens for child in token.children ]
                # if sentence is passive
                # if sentence is active transitive?
                if has_passive_deps(child_deps):
                    counts["pass"] += len(verb_tokens)
                    to_write += sent.text + " "
                elif has_active_deps(child_deps):
                    counts["act"] += len(verb_tokens)
                    chance = flip_coin(0.5)
                    if chance:
                        to_write += sent.text + " "
                        counts["kept"] += len(verb_tokens)
                    else:
                        random_token, start, end = random.choice(verb_tokens) # only replace 1 verb
                        replacement_verb = verb_forms[target_word][random_token.tag_]
                        replacement_sent = sent[:start].text_with_ws + replacement_verb + random_token.whitespace_ + sent[end:].text

                        to_write += replacement_sent + " "
                        removed.append(sent.text)
                        counts["changed"] += 1
                        counts["kept"] += len(verb_tokens) - 1
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
    parser.add_argument("--source_verb", "-s", type=str, help='verb whose frequency to emulate', default='last')
    parser.add_argument("--target_verb", "-t", type=str, help='verb to remove from dataset', default='hit')


    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir)
    source_verb = args.source_verb
    target_verb = args.target_verb

    nlp = load_model(require_gpu=True)
    matcher = load_matcher(nlp, [target_verb])

    with open(corpus_dir / "counts.json", 'r') as f:
        verb_counts = json.load(f)
    target_value = verb_counts[source_verb]["passive"]

    for filename in corpus_dir.glob("*.txt"):
        texts = []
        base_filename = filename.name
        print(f"Analyzing {base_filename}")
        with open(filename, 'r') as f:
            for line in f:
                texts.append(line.strip())
        save_folder = Path(args.target_dir) / f"{source_verb}_to_{target_verb}_swap"
        save_folder.mkdir(parents=True, exist_ok=True)
        save_file = save_folder / base_filename
        removed_file = save_folder / (filename.stem + "_removed.txt")
        counts = get_save_matches(save_file, matcher, source_verb, removed_file, texts)
        with open(save_folder / (filename.stem + "_counts.json"), 'w') as f:
            json.dump(counts,f)
        print("Done.")