import kenlm
import csv
import nltk
with open('/scratch/cl5625/exceptions/data/test_sentences.csv', mode ='r')as file:
    reader = csv.DictReader(file)
    data = list(reader)
    print(data)

model = kenlm.Model('bigram.binary')

active_scores = []
passive_scores = []
for line in data:
    active_scores.append(
        model.score(' '.join(nltk.word_tokenize(line['active'])), bos = True, eos = True)
    )
    passive_scores.append(
        model.score(' '.join(nltk.word_tokenize(line['passive'])), bos = True, eos = True)
    )

for old_dict, act_score, pass_score in zip(data, active_scores, passive_scores):
    old_list = list(old_dict.values())
    old_list.extend([act_score, pass_score, act_score-pass_score])
    print(old_list)

writer = csv.writer(open("/scratch/cl5625/exceptions/scores/bigram.csv", 'w'))
for old_dict, act_score, pass_score in zip(data, active_scores, passive_scores):
    row = list(old_dict.values())
    row.extend([act_score, pass_score, act_score-pass_score])
    print(row)
    writer.writerow(row)
