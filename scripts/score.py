import pyrootutils
from minicons import scorer
from itertools import product
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

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


def score_model(model: scorer.IncrementalLMScorer, df: pd.DataFrame):
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

def get_models():
    affectedness = ["low", "high"]
    seeds = ["495", "6910", "8397", "1208", "634"]
    amts = ["0", "10", "100", "500", "1000", "2000"]
    return [f"100M_{a}_{n}_{s}" for (a,s,n) in product(affectedness, seeds, amts)]


if __name__ == "__main__":
    failed = []
    test_path = PROJECT_ROOT / "decomp_dataset" / "gpt4o_generated" / "gpt_test.csv"
    df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained("craa/gpt2-with-test-verbs")
    model_names = get_models()
    for model_name in tqdm(model_names):
        try:
            model = AutoModelForCausalLM.from_pretrained(f"craa/{model_name}")
            model.resize_token_embeddings(len(tokenizer))
            model = scorer.IncrementalLMScorer(model=model, tokenizer=tokenizer, device="cpu")
            stimuli_df = score_model(model, df)
            stimuli_df.to_csv(PROJECT_ROOT / "scores" / "exp3" / f"{model_name}_scores.csv", index=False)
        except EnvironmentError:
            failed.append(model_name)
            continue

    print(failed)