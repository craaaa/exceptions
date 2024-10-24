from datetime import datetime
from openai import OpenAI
from tqdm import tqdm
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

client = OpenAI()

affected = False

prompt_file = "system_prompt_affected" if affected else "system_prompt_unaffected.txt"

with open(PROJECT_ROOT / "decomp_dataset" / prompt_file, "r") as f:
    system_prompt = f.read()

unaffected_verb_examples = """
    abut, believe, border, cost, dread, fear, fit, have, hear, know, lack, last, like, look like, miss, need, notice, overhear, receive, recognize, remember, resemble, respect, see, sense, sleep (i.e., hold enough sleeping space for), spot, total (i.e., add up to), trust, undergo, understand```
"""

affected_verb_examples = """
    bite, boil, break, burn, crush, demolish, drown, execute, exterminate, frighten, hammer, hit, kick, kill, knife, murder, punch, push, rob, shake, shoot, stab, strangle, suffocate, terrorize, thump, wreck
"""

examples = affected_verb_examples if affected else unaffected_verb_examples
save_location = "gpt4o_highaffected" if affected else "gpt4o_lowaffected"
for i in tqdm(range(50)):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Give me 15 examples. Some verbs you can use include: {examples}, but you can also use other verbs."
            }
        ]
    )
    x = completion.choices[0].message
    print(x)
    y = x.content
    try:
        json_part = y.split("```json")[1]
        json_part = json_part.split("```")[0]
        print(json_part)
        now = str(datetime.now().strftime("%y%m%d%H%M")) + ".json"

        with open(PROJECT_ROOT / "decomp_dataset" / save_location / now, "w") as f:
            f.write(json_part)
    except IndexError:
        print("Oops")