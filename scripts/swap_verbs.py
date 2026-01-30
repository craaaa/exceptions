import argparse
from corpus_utils import *
import json
import math
from tqdm import tqdm
import random
from datasets import load_dataset
from pathlib import Path


def get_save_matches(save_filename, replacement_word, target_matcher, mutating_matcher, removed_filename, changed_filename, texts, expected_num_sents_to_change, expected_num_sents_to_keep):
    counts = {
        "target": {"all": 0, "act": 0, "pass": 0, "other": 0, "kept": 0, "changed": 0}, 
        "mutating": {"all": 0, "act": 0, "pass": 0, "other": 0, "kept": 0, "removed": 0, "changed": 0},
    }
    removed = []
    changed = []
    with open(save_filename, 'w') as f:
        for doc in tqdm(nlp.pipe(texts, batch_size=1000), total=len(texts)):
            to_write = ""
            for sent in doc.sents:
                # if all removals/changes have been made, don't bother checking any more
                if counts["target"]["changed"] == expected_num_sents_to_change and counts["mutating"]["removed"] == expected_num_sents_to_change:
                    to_write += sent.text + " "
                    continue

                target_matches = target_matcher(sent)
                mutating_matches = mutating_matcher(sent)
                if not target_matches and not mutating_matches:
                    to_write += sent.text + " "
                    continue
                
                # First, check if the sentence contains the target verb
                if target_matches:
                    verb_tokens = [(token, start, end) for (_, start, end) in target_matches for token in sent[start:end] if token.pos_ == "VERB"]
                    counts["target"]["all"] += len(verb_tokens)
                    child_deps = [child.dep_ for (token,_,_) in verb_tokens for child in token.children ]
                    # if sentence is passive
                    # if sentence is active transitive?
                    if has_passive_deps(child_deps):
                        counts["target"]["pass"] += len(verb_tokens)
                        to_write += sent.text + " "
                    elif has_active_deps(child_deps):
                        counts["target"]["act"] += len(verb_tokens)
                        # If we've already changed enough sentences, leave the original sentence
                        if counts["target"]["changed"] >= expected_num_sents_to_change:
                            to_write += sent.text + " "
                            counts["target"]["kept"] += len(verb_tokens)
                        else:
                            # Otherwise, replace the target verb with the mutating verb
                            chosen_verb, start, end = random.choice(verb_tokens) # only replace 1 verb
                            replacement_verb = verb_forms[replacement_word][chosen_verb.tag_]
                            replacement_sent = sent[:start].text_with_ws + replacement_verb + chosen_verb.whitespace_ + sent[end:].text

                            to_write += replacement_sent + " "
                            changed.append(replacement_sent)
                            counts["target"]["changed"] += 1
                            counts["target"]["kept"] += len(verb_tokens) - 1
                    else:
                        counts["target"]["other"] += len(verb_tokens)
                        to_write += sent.text + " "
                elif mutating_matches:
                    # If the sentence contains the mutating verb and we haven't downsampled enough, remove and add it to the removed list
                    verb_tokens = [(token, start, end) for (_, start, end) in mutating_matches for token in sent[start:end] if token.pos_ == "VERB"]
                    counts["mutating"]["all"] += len(verb_tokens)
                    child_deps = [child.dep_ for (token,_,_) in verb_tokens for child in token.children ]
                    if has_passive_deps(child_deps):
                        counts["mutating"]["pass"] += len(verb_tokens)
                    elif has_active_deps(child_deps):
                        counts["mutating"]["act"] += len(verb_tokens)
                        # If we've already removed enough sentences, leave the original sentence
                        if counts["mutating"]["removed"] >= expected_num_sents_to_change:
                            to_write += sent.text + " "
                            counts["mutating"]["kept"] += len(verb_tokens)
                        else:
                            # Otherwise, remove the mutating verb
                            removed.append(sent.text)
                            counts["mutating"]["removed"] += len(verb_tokens)
                    else:
                        counts["mutating"]["other"] += len(verb_tokens)
                        to_write += sent.text + " "
                else:
                    print("Sentence contains neither target nor mutating verb?")
                    print(sent.text)
                    exit(1)
            
            f.write(to_write.strip() + "\n")
            # all sentences are accounted for
            assert counts["target"]["all"] == counts["target"]["act"] + counts["target"]["pass"] + counts["target"]["other"]
            assert counts["mutating"]["all"] == counts["mutating"]["act"] + counts["mutating"]["pass"] + counts["mutating"]["other"]

            # all active sentences are accounted for
            assert counts["target"]["act"] == counts["target"]["kept"] + counts["target"]["changed"]
            assert counts["mutating"]["act"] == counts["mutating"]["kept"] + counts["mutating"]["removed"]
                    
    print(f"Saved dataset to {save_filename}.")
    with open(removed_filename, 'w') as f:
        for line in removed:
            f.write(line)
            f.write("\n")
    with open(changed_filename, 'w') as f:
        for line in changed:
            f.write(line)
            f.write("\n")
    return counts



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into chunks")
    parser.add_argument("--corpus_dir", type=str, help='prefix of path to corpus', default='/scratch/cl5625/exceptions/data/100M')
    parser.add_argument("--target_dir", type=str, help='directory to save dataset', default='/scratch/cl5625/exceptions/data/')
    parser.add_argument("--source_verb", "-s", type=str, help='mutating verb', default='last')
    parser.add_argument("--target_verb", "-t", type=str, help='target verb', default='hit')
    parser.add_argument("--ratio", "-r", type=float, help='expected ratio of mutating verb sentences to use target verb contexts (between 0 and 1)', default=0.3)

    args = parser.parse_args()
    corpus_dir = Path(args.corpus_dir)
    mutating_verb = args.source_verb
    target_verb = args.target_verb

    nlp = load_model(require_gpu=True)
    target_matcher = load_matcher(nlp, [target_verb])
    mutating_matcher = load_matcher(nlp, [mutating_verb])

    # load train and validation files from huggingface dataset
    dataset = load_dataset("craa/exceptions", "default")
    # Load verb counts
    with open(corpus_dir / "counts.json", 'r') as f:
        verb_counts = json.load(f)
    
    target_active = verb_counts[target_verb]["active"]
    mutating_active = verb_counts[mutating_verb]["active"]
    
    assert 0 <= args.ratio <= 1, f"Ratio must be between 0 and 1, got {args.ratio}"
    expected_num_sents_to_change = int(math.ceil(mutating_active * args.ratio))
    expected_num_sents_to_keep = int(math.floor(mutating_active - expected_num_sents_to_change))

    # make sure not to change the total number of mutating verb sentences
    assert expected_num_sents_to_keep + expected_num_sents_to_change == mutating_active

    # make sure there are enough target verb sentences to change
    assert expected_num_sents_to_change <= target_active, f"Expected {expected_num_sents_to_change} sentences to change, but only {target_active} sentences of {target_verb} are available."

    print(f"Dataset contains: {mutating_active} active sentences of {mutating_verb} and {target_active} active sentences of {target_verb}.")
    print(f"Changing {expected_num_sents_to_change} sentences containing {mutating_verb} to sentences containing {target_verb} and keeping {expected_num_sents_to_keep} sentences containing {mutating_verb}.")

    for split in ["train", "validation"]:
        print(f"Analyzing {split}")
        dataset_split = dataset[split].shuffle(seed=4812)
        texts = dataset_split["text"]

        save_folder = Path(args.target_dir) / f"swap_{args.ratio}_{mutating_verb}_to_{target_verb}"
        save_folder.mkdir(parents=True, exist_ok=True)

        counts = get_save_matches(
            save_filename=save_folder / f"{split}.txt",
            replacement_word=mutating_verb,
            target_matcher=target_matcher,
            mutating_matcher=mutating_matcher,
            removed_filename=save_folder / f"{split}_removed.txt",
            changed_filename=save_folder / f"{split}_changed.txt",
            texts=texts,
            expected_num_sents_to_change=expected_num_sents_to_change,
            expected_num_sents_to_keep=expected_num_sents_to_keep)

        with open(save_folder / f"{split}_counts.json", 'w') as f:
            json.dump(counts,f)
        print("Done.")