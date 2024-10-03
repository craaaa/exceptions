from collections import Counter
from dotenv import load_dotenv
from minicons import scorer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


participles = {
    "froze": "frozen",
    "sang": "sung",
    "drank": "drunk",
    "broke": "broken",
    "tore": "torn",
    "began": "begun",
    "ate": "eaten",
    "took": "taken",
    "stole": "stolen",
    "shook": "shaken",
    "was": "been",
    "drove": "driven",
    "draw": "drawn",
    "wrote": "written",
    "grew": "grown",
    "sewed": "sewn",
    "beat": "beaten",
    "take": "taken",
    "chose": "chosen",
    "hid": "hidden",
    "threw": "thrown",
    "bit": "bitten",
    "forgave": "forgiven",
    "spoke": "spoken",
    "drew": "drawn",
    "slayed": "slain",
    "saw": "seen",
    "knew": "known",
    "forgot": "forgotten",
    "underwent": "undergone",
    "got": "gotten",
    "proved": "proven",
    "rang": "rung",
}


def switch_verbs(word_list):
    sources = list(participles.keys())
    new_list = word_list

    for i, word in enumerate(word_list):
        if word in sources:
            new_list[i] = participles[word]
    return new_list


def get_matches(active_words, passive_words):
    matches = []
    for a, p in zip(active_words, passive_words):
        singular = Counter(switch_verbs(a) + ["was", "by"])
        plural = Counter(switch_verbs(a) + ["were", "by"])
        passive = Counter(p)
        matches.append(singular == passive or plural == passive)

    return matches


def check_correspondence(generated_sentences):
    # check that passive and active contain the same words
    active_words = generated_sentences["active"].apply(
        lambda x: x.lower().strip(" .").split()
    )
    passive_words = generated_sentences["passive"].apply(
        lambda x: x.lower().strip(" .").split()
    )
    return get_matches(active_words, passive_words)
    # [
    #     Counter(a + ["was", "by"]) == Counter(p)
    #     or Counter(a + ["were", "by"]) == Counter(p)
    #     for (a, p) in zip(active_words, passive_words)
    # ]


# sample n sentences if more than n are available; if not, take all
def sample_n(df, n=5):
    if len(df) >= n:
        return df.sample(n=n, random_state=1)
    return df


def sample_from_filtered(filtered_file):
    load_dotenv()

    generated_sentences = pd.read_csv(
        filtered_file,
        low_memory=False,
    )

    generated_sentences["match"] = check_correspondence(generated_sentences)

    # sample 3 sentences from list
    filtered = generated_sentences.loc[generated_sentences["bad"].isna()]
    sampled_5 = filtered.groupby("verb", group_keys=True).apply(sample_n)

    # remove sentences for which the verb is strange
    excluded_verbs = generated_sentences.loc[
        generated_sentences["exclude"].notna()
    ].verb.unique()

    sampled_5 = sampled_5[~sampled_5["verb"].isin(excluded_verbs)].reset_index(
        drop=True
    )
    return sampled_5


def get_ambridge_verbs(ambridge_verb_file):
    ambridge_verbs = pd.read_csv(ambridge_verb_file, low_memory=False)

    ambridge_verbs = ambridge_verbs.drop(
        columns=[
            "Participant",
            "Set",
            "FullSet",
            "Direction",
            "Agegroup",
            "Age",
            "Tno",
            "Rev",
            "Stype",
            "Agent",
            "Testorder",
            "Sheetorder",
            "Rating",
        ]
    )
    ambridge_verbs = ambridge_verbs.drop_duplicates()
    return ambridge_verbs


def get_test_sentences():
    samples = sample_from_filtered(
        PROJECT_ROOT / "data" / "ambridge" / "generated_filtered_fixed.csv",
    )
    samples["verb"] = samples["verb"].astype(pd.StringDtype())

    ambridge_verbs = get_ambridge_verbs(
        PROJECT_ROOT / "data" / "ambridge" / "01PassiveJud.csv"
    )
    ambridge_verbs["Verb"] = ambridge_verbs["Verb"].astype(pd.StringDtype())

    joined = pd.DataFrame(samples).merge(
        right=ambridge_verbs, left_on="verb", right_on="Verb", how="left"
    )
    print(joined)

    joined.to_csv(
        PROJECT_ROOT / "data" / "ambridge" / "ambridge_test_sentences.csv", index=False
    )


if __name__ == "__main__":
    get_test_sentences()

# TODO: Get scores for each of the 5 models


# TODO: map each verb to its Ambridge "verb class"
# sampled_5.to_csv(PROJECT_ROOT / "data" / "ambridge" / "generated_final.csv")
