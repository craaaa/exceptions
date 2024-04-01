from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"
model = AutoModelForCausalLM.from_pretrained("models/100M_1/checkpoint-128400", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("gpt2", padding_side="left")

model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(device)
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

model_inputs = tokenizer(
    ["A list of colors: red, blue", "Portugal is", "The"], return_tensors="pt", padding=True
).to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=True, top_k=50, num_beams=5)

generations = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(generations)