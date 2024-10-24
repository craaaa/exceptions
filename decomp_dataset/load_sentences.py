from collections import Counter
import csv
from datetime import datetime
import json
import spacy
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

nlp = spacy.load("en_core_web_trf")

def check_validity(sample, affectedness="low"):
    def process_string(string):
        return string.lower().replace("’", "'").replace(".","") 
    
    if process_string(sample['sentence']) not in process_string(sample['paragraph']):
        return False
    
    if not all(isinstance(n, int) for n in sample['scores']):
        return False

    if affectedness == "low":
        if sum(sample['scores']) > 18:
            return False
    elif affectedness == "high":
        if sum(sample['scores']) < 30:
            return False
    else:
        print("affectedness should be 'high' or 'low'")
    
    verb = nlp(sample["verb"])[0]
    doc = nlp(sample["sentence"])

    if sample['verb'] in sample['sentence'].split(" "):
        verb_in_sent = next(x for x in doc if x.text == verb.text)
    elif verb.lemma_ in [x.lemma_ for x in doc]:
        verb_in_sent = next(x for x in doc if x.lemma_ == verb.lemma_)
        sample['verb'] = verb_in_sent.text
    else: 
        print("ohno")
        print(sample)
        return False

    morphology = verb_in_sent.morph
    novel_verb_form = f"NOVEL_VERB_{morphology}"
    novel_verb_forms.add(novel_verb_form)
    sample['sentence_novel'] = sample['sentence'].replace(sample['verb'], novel_verb_form)
    sample['paragraph_novel'] = sample['paragraph'].replace(sample['sentence'], sample['sentence_novel'])

    return set(sample.keys()) == set([
                "subject",
                "object",
                "verb",
                "sentence",
                "sentence_novel",
                "paragraph",
                "paragraph_novel",
                "scores",
    ])

generated_jsons = PROJECT_ROOT / "decomp_dataset" / "gpt4o_generated"

samples = []
novel_verb_forms = set()

for dir in generated_jsons.glob('**/'):
    if dir.name == "high" or dir.name == "low":
        affectedness = dir.name
        for filepath in dir.iterdir():
            print(filepath)
            with open(filepath, "r") as f:
                x = json.load(f)
                for sample in x:
                    if "ratings" in sample.keys():
                        sample["scores"] = sample["ratings"]
                        sample.pop("ratings")
                    if check_validity(sample, affectedness=affectedness):
                        sample['affectedness'] = affectedness
                        samples.append(sample)

y = [x["verb"] for x in samples]
# print(Counter(y))

y = [x["sentence_novel"] for x in samples]

print(len(y))
# print([(z, w) for (z, w) in Counter(y).items() if w > 1])
print(y)

now = str(datetime.now().strftime("%y%m%d%H%M"))

keys = samples[0].keys()

with open(PROJECT_ROOT / "decomp_dataset" / "gpt4o_generated" / f"compiled_{now}.csv", 'w', newline='') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(samples)


print(samples[:3])
print(novel_verb_forms)