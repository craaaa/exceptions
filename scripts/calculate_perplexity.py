from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

models = [
    "openai-community/gpt2",
    "/scratch/cl5625/exceptions/models/100M_1/checkpoint-128400",
    "/scratch/cl5625/exceptions/models/100M_2/checkpoint-163200",
    "/scratch/cl5625/exceptions/models/100M_3/checkpoint-166800",
    "/scratch/cl5625/exceptions/models/100M_4/checkpoint-166800",
    "/scratch/cl5625/exceptions/models/100M_5/checkpoint-166800",
    # "/scratch/cl5625/exceptions/models/replace_cost_with_carry_40_3502/checkpoint-123600",
    # "/scratch/cl5625/exceptions/models/replace_cost_with_carry_40_3519/checkpoint-52800",
    # "/scratch/cl5625/exceptions/models/replace_cost_with_carry_40_3553/checkpoint-51600",
    # "/scratch/cl5625/exceptions/models/replace_cost_with_carry_40_3587/checkpoint-62400",
    # "/scratch/cl5625/exceptions/models/replace_last_with_carry_40_3502/checkpoint-117600",
    # "/scratch/cl5625/exceptions/models/replace_last_with_carry_40_3519/checkpoint-117600",
    # "/scratch/cl5625/exceptions/models/replace_last_with_carry_40_3536/checkpoint-120000",
    # "/scratch/cl5625/exceptions/models/replace_last_with_carry_40_3553/checkpoint-120000",
    # "/scratch/cl5625/exceptions/models/replace_last_with_carry_40_3587/checkpoint-120000",
]

device = "cuda"
test = load_dataset("text", data_files="/scratch/cl5625/exceptions/data/100M/validation.txt",split="train")
tokenizer = GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

perplexities = {}
for model_id in models:
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device)

    max_length = model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    perplexities[model_id] = ppl.item()

for name, ppl in perplexities.items():
    print(name + "\t\t\t" + str(ppl))