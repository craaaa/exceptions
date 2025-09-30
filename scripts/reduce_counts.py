from corpus_utils import *
from fire import Fire
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

random.seed(42)

def get_save_matches(save_filename, removed_filename, texts, target_verb, target_words=100_000_000):
    nlp = load_model(require_gpu=True)
    matcher = load_matcher(nlp, [target_verb])

    counts = {"all": 0, "act": 0, "pass": 0, "other": 0, "kept": 0, "removed": 0}
    removed = []
    num_words = 0
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
                if has_passive_deps(child_deps):
                    counts["pass"] += len(verb_tokens)
                    removed.append(sent.text)
                    counts["removed"] += len(verb_tokens)
                # if sentence is active transitive?
                elif has_active_deps(child_deps):
                    counts["act"] += len(verb_tokens)
                    to_write += sent.text + " "
                else:
                    counts["other"] += len(verb_tokens)
                    to_write += sent.text + " "
            f.write(to_write.strip() + "\n")
            num_words += len(to_write.split())
    print(f"Saved dataset to {save_filename}.")
    with open(removed_filename, 'w') as f:
        for line in removed:
            f.write(line)
            f.write("\n")
    
    print(f"Total words kept: {num_words}.")
    if num_words < target_words:
        print(f"WARNING: Total words kept ({num_words}) is less than target value ({target_words}).")

    return counts

def upload_to_huggingface(dataset, dataset_name, save_filename):
    new_subset = load_dataset("text", data_files=save_filename)
    subset_name = os.path.basename(save_filename)
    dataset.filter(lambda x: len(x["text"]) > 0)
    print(len(dataset))
    dataset[subset_name] = new_subset
    dataset.push_to_hub(dataset_name)

def main(dataset_name='craa/100M',
        counts_filename='results/corpus/100M/counts.json', 
        source_verb='last', # verb whose frequency to emulate
        target_verb='hit',
        chunk=0): # verb to remove from dataset
    with open(PROJECT_ROOT / counts_filename, 'r') as f:
        verb_counts = json.load(f)
    target_value = verb_counts[source_verb]["passive"]
    print(f"Removing passives of {target_verb} to reach target value of {source_verb}: {target_value}")

    dataset = load_dataset(dataset_name)
    dataset_chunk_size = int(len(dataset['train']['text']) / 10)
    if chunk == 9:
        texts = dataset['train']['text'][chunk * dataset_chunk_size:]
    else:
        texts = dataset['train']['text'][chunk * dataset_chunk_size:(chunk + 1) * dataset_chunk_size]

    local_dataset_name = source_verb + "_to_" + target_verb + "_frequency" + "_" + str(chunk)
    save_folder = PROJECT_ROOT / "results" / "corpus" / local_dataset_name
    save_folder.mkdir(parents=True, exist_ok=True)
    print(f"Analyzing {local_dataset_name}")

    corpus_file = save_folder / (local_dataset_name + ".txt")
    removed_file = save_folder / (local_dataset_name + "_removed.txt")

    counts = get_save_matches(corpus_file, removed_file, texts, target_verb)

    with open(save_folder / (local_dataset_name + "_counts.json"), 'w') as f:
        json.dump(counts,f)
    print("Filtered dataset.")

    # Upload new dataset to huggingface
    # upload_to_huggingface(dataset, dataset_name, corpus_file)
    # print(f"Updated dataset {dataset_name} on huggingface.")

if __name__ == "__main__":
    Fire(main)