import random
import spacy
import spacy_transformers
from spacy.matcher import Matcher

random.seed(4182)

cost_verbs = ["earn", "cost", "fetch"]
take_verbs = ["take", "need", "last", "require"]
benefit_verbs = ["benefit", "help", "profit", "strengthen"]
ooze_verbs = ["discharge", "emanate", "emit", "radiate"]
match_verbs = ["approximate", "match", "mirror", "resemble"]
agent_patient = ["hit", "wash", "carry", "push", "drop"]
exp_theme = ["see", "hear", "like", "know", "remember"]

verb_forms = {
    'last': {
        'VB': 'last',
        'VBD': 'lasted',
        'VBG': 'lasting',
        'VBN': 'lasted',
        'VBP': 'last',
        'VBZ': 'lasts',
    },
    'take': {
        'VB': 'take',
        'VBD': 'took',
        'VBG': 'taking',
        'VBN': 'taken',
        'VBP': 'take',
        'VBZ': 'takes',
    },
    'require': {
        'VB': 'require',
        'VBD': 'required',
        'VBG': 'requiring',
        'VBN': 'required',
        'VBP': 'require',
        'VBZ': 'requires',
    }
}

def load_model(require_gpu=True):
    if require_gpu:
        spacy.require_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner", "textcat", "entity_linker", "entity_ruler", "textcat_multilabel", "senter", "sentencizer", "tok2vec"])
    return nlp
    
def load_matcher(model, target_lemmas):
    matcher = Matcher(model.vocab)
    pattern = [{"LEMMA": {"IN": target_lemmas}}]
    matcher.add("target_verbs", [pattern])
    return matcher

def has_passive_deps(deps):
    return "nsubjpass" in deps or "csubjpass" in deps or "auxpass" in deps

def has_active_deps(deps):
    return "nsubj" in deps and ("dobj" in deps or "ccomp" in deps) and "dative" not in deps

def flip_coin(prob=0.5):
    random.random() < prob

