from datasets import concatenate_datasets, load_dataset

split_to_use = "low_100"

high_split = load_dataset("craa/100M", split="high_2000")
low_split = load_dataset("craa/100M", split="low_2000")

# Get first 2000 lines of each split
high_base = high_split.select(range(2000))
low_base = low_split.select(range(2000))

# Get last 2000 lines of each split
high_novel = high_split.select(range(len(high_split) - 2000, len(high_split)))
low_novel = low_split.select(range(len(low_split) - 2000, len(low_split)))

novel_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1500]
splits = {
    "novel": {"high": high_novel, "low": low_novel},
    "base": {"high": high_base, "low": low_base},
}

for novel_value in novel_values:
    for (split_type, other_type) in [("high", "low"), ("low", "high")]:
        print(
            f"Creating dataset with {novel_value} {split_type}-affectedness novel sentences"
        )

        novel_sentences = splits["novel"][split_type].select(range(novel_value))
        base_sentences = splits["base"][split_type].select(range(2000 - novel_value))
        other_sentences = splits["base"][other_type].select(range(2000))

        dataset = concatenate_datasets([other_sentences, base_sentences, novel_sentences])
        print(dataset["text"][0])
        print(dataset["text"][-1])

        dataset.push_to_hub("craa/100M", split=f"{split_type}_{novel_value}")
        # dataset = concatenate_datasets([base, novel])
        # dataset.push_to_hub(f"100M_{split_type}_{novel_value}")
# dataset = concatenate_datasets([train_split, split_to_use])
