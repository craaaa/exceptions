from typing import Optional
from tqdm import tqdm
import pyrootutils
from pathlib import Path
from itertools import product
import argparse

from pandas import DataFrame, read_csv
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from minicons import scorer

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

GOOD_SENTENCE_COLNAME = "active"
BAD_SENTENCE_COLNAME = "passive"


def get_stim_from_df(df):
    good_sentences = list(df[GOOD_SENTENCE_COLNAME])
    bad_sentences = list(df[BAD_SENTENCE_COLNAME])
    stimuli = list(zip(good_sentences, bad_sentences))

    return stimuli


def score_model(model: scorer.IncrementalLMScorer, df: DataFrame):
    stimuli = get_stim_from_df(df)
    stimuli_dl = DataLoader(stimuli, batch_size=20)

    active_scores = []
    passive_scores = []
    for batch in stimuli_dl:
        good, bad = batch
        good_scores = model.sequence_score(good, reduction=lambda x: x.sum(0).item())
        bad_scores = model.sequence_score(bad, reduction=lambda x: x.sum(0).item())
        active_scores.extend(good_scores)
        passive_scores.extend(bad_scores)

    df["active_score"] = active_scores
    df["passive_score"] = passive_scores
    df["pass_drop"] = df["active_score"] - df["passive_score"]
    return df

def get_exp3_models():
    affectedness = ["low", "high"]
    affectedness = ["low"]
    seeds = ["495", "6910", "8397", "1208", "634"]
    seeds = ["495"]
    amts = ["0","1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1500", "10", "100", "1000", "2000"]
    amts = ["0"]
    return [f"{a}_{n}_{s}" for (a,s,n) in product(affectedness, seeds, amts)]

def get_exp2a_models():
    target_verb = ["last", "cost", "resemble"]
    mutating_verb = ["drop", "carry", "push", "hit"]
    seeds = ["1001", "2128", "3591", "5039", "40817"]
    seeds = ["40817"]
    return [f"{t}_to_{m}_frequency_{s}" for (t,m,s) in product(target_verb, mutating_verb, seeds)]

def get_exp1b_models():
    seeds = ["1001", "2128", "3591", "5039", "40817"]
    return [f"100M_{s}" for s in seeds]

def get_latest_checkpoint(model_path: Path):
    checkpoints = [p for p in model_path.glob("checkpoint-*") if p.is_dir()]
    return sorted(checkpoints)[-1]

def main(exp: str, test_sentence_path: Optional[Path] = None):
    if exp == "3":
        tokenizer = AutoTokenizer.from_pretrained("craa/gpt2-with-test-verbs")
        if test_sentence_path is None:
            test_sentence_path = PROJECT_ROOT / "data" / "exp3_test_sentences.csv"
        model_names = get_exp3_models()
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if test_sentence_path is None:
            test_sentence_path = PROJECT_ROOT / "data" / "test_sentences.csv"
        if exp == "1b":
            model_names = get_exp1b_models()
        elif exp == "2a":
            model_names = get_exp2a_models()
        else:
            raise ValueError(f"Invalid experiment number: {exp}")

    df = read_csv(test_sentence_path)
    failed = []
    df['active'] = df.apply(lambda row: row['active'] + ".", axis=1)
    df['passive'] = df.apply(lambda row: row['passive'] + ".", axis=1)

    for model_name in tqdm(model_names):
        print(f"Getting scores from {model_name}")
        try:
            model_path_string = str(PROJECT_ROOT / "models" / f"{model_name}" / get_latest_checkpoint(PROJECT_ROOT / "models" / f"{model_name}"))
            print(model_path_string)
            print(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_path_string, local_files_only=True)
            model.resize_token_embeddings(len(tokenizer))
            model = scorer.IncrementalLMScorer(model=model, tokenizer=tokenizer, device="cpu")
                        
            stimuli_df = score_model(model, df)

            stimuli_df.to_csv(PROJECT_ROOT / "scores" / f"exp{exp}" / f"{model_name}.csv", index=False)
        except Exception as e:
            print(f"Error scoring {model_name}: {e}")
            failed.append(model_name)
            continue

    if failed:
        print(f"Failed to score {len(failed)} models:")
        print(failed)

if __name__ == "__main__":
    default_test_path = PROJECT_ROOT / "data" / "test_sentences.csv"

    parser = argparse.ArgumentParser(description="Score experiment models.")
    parser.add_argument("--exp", type=str, default="1b", required=False, help="Experiment number (e.g., 1b, 2a, 3)")
    parser.add_argument("--test_sentence_path", type=str, default=str(default_test_path), required=False, help="Path to custom test sentences CSV")
    args = parser.parse_args()

    main(exp=args.exp, test_sentence_path=Path(args.test_sentence_path))
