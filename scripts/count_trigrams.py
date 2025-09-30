from collections import Counter
from itertools import chain
from nltk import ngrams
from tqdm import tqdm
from transformers import AutoTokenizer
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

# Set n
n = 3

tokenizer = AutoTokenizer.from_pretrained("craa/gpt2-with-test-verbs")

def main(use_tokenizer=True):
    with open(PROJECT_ROOT / "data" / "100M" / "train_100M.txt") as f:
        docs = f.readlines()

    # Tokenize documents; use tokenizer if True, else split on spaces
    tokenize_function = tokenizer.tokenize if use_tokenizer else lambda x: x.split()
    tokenized_docs = map(tokenize_function, tqdm(docs))

    # Get trigrams
    def get_trigrams(x):
        return list(ngrams(x, 3))

    all_trigrams = map(get_trigrams, tqdm(tokenized_docs))
    all_trigrams = chain.from_iterable(all_trigrams) # flatten
    # Count trigrams
    trigram_counts = Counter(all_trigrams)

    # Save trigram counts to csv
    with open(PROJECT_ROOT / "data" / "100M" / "trigram_counts.csv", "w") as f:
        for trigram, count in trigram_counts.items():
            string_trigram = tokenizer.convert_tokens_to_string(trigram)
            f.write(f"\"{string_trigram}\",{count}\n")

if __name__ == "__main__":
    main(use_tokenizer=False)
