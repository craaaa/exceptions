import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count number of tokens in a dataset.")
    parser.add_argument("--data_folder", type=str, help='Path to corpus.', default='/scratch/cl5625/exceptions/data/')
    parser.add_argument("--dataset_name", type=str, help='Path to corpus.', default='/scratch/cl5625/exceptions/data/100M/train_100M.txt')

    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    dataset_name = args.dataset_name
    dataset_dir = data_folder / dataset_name
    train_file = str(Path(dataset_dir / "train_100M.txt"))
    val_file = str(Path(dataset_dir / "validation.txt"))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    dataset = load_dataset("text", data_files={"train": train_file, "test": val_file})
    dataset = dataset.map(lambda examples: tokenizer(examples["text"]), batched=True)
    train_count = sum([len(ex['input_ids']) for ex in dataset['train']])
    val_count = sum([len(ex['input_ids']) for ex in dataset['test']])

    with open(dataset_dir / "token_counts.txt", 'w')as f:
        f.write(f"Train: {train_count} tokens\n")
        f.write(f"Train: {val_count} tokens\n")