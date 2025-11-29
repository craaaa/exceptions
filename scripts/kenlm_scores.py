import kenlm
import csv
import nltk
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

#test_file = '/scratch/cl5625/exceptions/data/test_sentences.csv'
test_file = PROJECT_ROOT / "data" / "ambridge" / "ambridge_test_sentences.csv"
model_name = "trigram"

with open(test_file, mode ='r')as file:
    reader = csv.DictReader(file)
    data = list(reader)
    print(data)

model = kenlm.Model(str(PROJECT_ROOT / "models"/ f"{model_name}.binary"))

active_scores = []
passive_scores = []
active_unks = []
passive_unks = []
active_tokens = []
passive_tokens = []
for line in data:
    active_sent_tokenized = ' '.join(nltk.word_tokenize(line['active']))
    passive_sent_tokenized = ' '.join(nltk.word_tokenize(line['passive']))
    active_scores.append(
        model.score(active_sent_tokenized, bos = True, eos = True)
    )
    passive_scores.append(
        model.score(passive_sent_tokenized, bos = True, eos = True)
    )
    active_unks.append(sum(1 for (_, _, oov) in model.full_scores(active_sent_tokenized) if oov))
    passive_unks.append(sum(1 for (_, _, oov) in model.full_scores(passive_sent_tokenized) if oov))

    active_tokens.append(sum(1 for (_, _, oov) in model.full_scores(active_sent_tokenized)))
    passive_tokens.append(sum(1 for (_, _, oov) in model.full_scores(passive_sent_tokenized)))

for old_dict, act_score, pass_score, act_unks, pass_unks in zip(data, active_scores, passive_scores, active_unks, passive_unks):
    old_list = list(old_dict.values())
    old_list.extend([act_score, pass_score, act_score-pass_score, act_unks, pass_unks])
    print(old_list)

writer = csv.writer(open(PROJECT_ROOT / "scores" / "exp1b" / f"really_{model_name}.csv", 'w'))
for old_dict, act_score, pass_score, act_unks, pass_unks in zip(data, active_scores, passive_scores, active_unks, passive_unks):
    row = list(old_dict.values())
    row.extend([act_score, pass_score, act_score-pass_score, act_unks, pass_unks])
    print(row)
    writer.writerow(row)

print("Average active unks: ", sum(active_unks)/len(active_unks))
print("Average passive unks: ", sum(passive_unks)/len(passive_unks))
print("Average active tokens: ", sum(active_tokens)/len(active_tokens))
print("Average passive tokens: ", sum(passive_tokens)/len(passive_tokens))

print("% tokens unk: ", sum(active_unks)/sum(active_tokens))
print("% tokens unk: ", sum(passive_unks)/sum(passive_tokens))