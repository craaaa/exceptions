#!/usr/bin/zsh

myarray=("${(@f)$(< test_sentences_text.txt)}")
for item in $myarray 
do
echo $item
grep -oiF $item 100M/train_100M.txt | wc -l
done