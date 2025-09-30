import argparse
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP
from transformers.tokenization_utils import BatchEncoding
from typing import List, Tuple
import pandas as pd

def load_model(
    device: str, init_from: str="huggingface"
) -> GPT2LMHeadModel:
    if init_from == 'huggingface':
        model = GPT2LMHeadModel.from_pretrained("gpt2") # vanilla GPT2
    else:
        model = GPT2LMHeadModel.from_pretrained(checkpoint)
        # We need to resize the embedding layer because we added the pad token.
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        model.to(device)
    
    return model

def _add_special_tokens(text: str) -> str:
    return tokenizer.bos_token + text + tokenizer.eos_token

def _tokens_log_prob_for_batch(
    text: List[str]
) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:

    outputs: List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]] = []
    if len(text) == 0:
        return outputs

    text = list(map(_add_special_tokens, text))
    encoding: BatchEncoding = tokenizer.batch_encode_plus(
        text, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        ids = encoding["input_ids"].to(model.device)
        attention_mask = encoding["attention_mask"].to(model.device)
        nopad_mask = attention_mask
        logits: torch.Tensor = model(ids, attention_mask=attention_mask)[0]

    for sent_index in tqdm(range(5)):
        sent_nopad_mask = nopad_mask[sent_index] 
        # len(tokens) = len(text[sent_index]) + 1 
        sent_tokens = [tok for tok, mask in zip(tokenizer.convert_ids_to_tokens(ids[sent_index]),sent_nopad_mask) if mask][1:] # don't include BOS token
        print(sent_tokens)

        # sent_ids.shape = [len(text[sent_index]) + 1]
        sent_ids = ids[sent_index][1:len(sent_tokens)+1,]
        # logits.shape = [len(text[sent_index]) + 1, vocab_size]
        sent_logits = logits[sent_index][1:len(sent_tokens)+1,]
        # ids_scores.shape = [seq_len + 1]
        sent_ids_scores = sent_logits.gather(1, sent_ids.unsqueeze(1)).squeeze(1)
        print(f"ID scores: {sent_ids_scores}")
        # log_prob.shape = [seq_len + 1]
        sent_log_probs = sent_ids_scores - sent_logits.logsumexp(axis=1) #softmax?
        print(f"Log probs: {sent_log_probs}")
        
        sent_log_probs = sent_log_probs.float()
        print(f"Log probs: {sent_log_probs}")
        sent_ids = sent_ids.float()
        
        output = (sent_log_probs, sent_ids, sent_tokens)
        outputs.append(output)

    return outputs

def reduce(x: Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]
) -> Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]:

    sentence_length = torch.tensor(x[0].shape[0])
    sentence_score = x[0].logsumexp(0)
    print(x[0])
    print(x[0].logsumexp(0))
    normalized_sentence_score = sentence_score - torch.log(sentence_length)

    return (sentence_score.item(), x[1], x[2])

def get_sentence_scores(text: List[str]
) -> List[Tuple[torch.DoubleTensor, torch.LongTensor, List[str]]]:

    batch_log_probs = _tokens_log_prob_for_batch(text)
    sentence_scores = map(reduce, batch_log_probs)
    return list(sentence_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a model on test sentences.')
    parser.add_argument('testfile', default='data/test_sentences.csv')
    parser.add_argument('-m', '--model', type=str, 
                        help="Location of model to test", default="100M_4")
    parser.add_argument('-c', '--checkpoint', type=str,
                        help="Checkpoint number of model to test", default='168000')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<|pad|>"]})
    tokenizer.pad_token="<|pad|>"

    model_name = args.model
    checkpoint_number = args.checkpoint
    checkpoint = model_name
   
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        print("CUDA and MPS not available!")

    model = load_model(device, model_name)

    test_sentences = pd.read_csv(args.testfile)
    active_scores = get_sentence_scores(test_sentences['active'].tolist())
    test_sentences['active_score'] = list(map(lambda x: x[0], active_scores))

    passive_scores = get_sentence_scores(test_sentences['passive'].tolist())
    test_sentences['passive_score'] = list(map(lambda x: x[0], passive_scores))

    test_sentences['pass_drop'] = test_sentences['active_score'] - test_sentences['passive_score']

    test_sentences.to_csv(f'scores/test_sentences_scored_x.csv')
