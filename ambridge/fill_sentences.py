from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import pyrootutils
import re

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

load_dotenv()

ambridge_judgments = pd.read_csv(PROJECT_ROOT / "data" / "01PassiveJud.csv")
ambridge_verbs = ambridge_judgments.Verb.unique()

print(sorted(ambridge_verbs))
client = OpenAI()

generation_prompt = """
I will give you a verb. Create ten transitive sentences in the past tense using the verb.\
    The sentences should be in the form [A] [verb] [B] with as little additional content as possible. Do not use pronouns or relative pronouns.\n
Verb: eat\n
Sentences:\n
1. The mouse ate some cheese.\n
2. A man ate a bagel.\n
3. My brother ate the pasta.\n
\n
Verb: {}\n
Sentences:
"""


bleached_generation_prompt = """
I will give you a verb. Make two sentences using the verb where you can only use indefinite pronouns "somebody", "something", "someone", "somewhere", etc.\n
\n
Verb: eat\n
1. Somebody ate something.\n
2. Something ate someone.\n
\n
Verb: {}\n
Sentences:\n
"""

passivize_prompt = """
Passivize the given sentence.
Input: The mouse ate the cheese.
Output: The cheese was eaten by the mouse.

Input: {}
Output: 
"""

active = []
passive = []
verb = []
for test_verb in tqdm(ambridge_verbs):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": bleached_generation_prompt.format(test_verb)},
        ],
    )

    message = completion.choices[0].message.content + "\n"
    active_sentences = re.findall(r"\d. (.*)\n", message)

    passivized_sentences = []
    for sent in active_sentences:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": passivize_prompt.format(sent)},
            ],
        )
        passivized_sentences.append(completion.choices[0].message.content)

    if len(active_sentences) != len(passivized_sentences):
        print("Error!!")
        print(active_sentences)
        print(passivized_sentences)
        break
    
    active.extend(active_sentences)
    passive.extend(passivized_sentences)
    verb.extend([test_verb] * len(active_sentences))

print(active)
print(passive)
print(verb)

df = pd.DataFrame(
    {'active': active,
     'passive': passive,
     'verb': verb}
)

df.to_csv(PROJECT_ROOT / "data" / "bleached_generated.csv", index=False)