from lemminflect import getAllInflectionsOOV
import pandas as pd
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

x = pd.read_csv(PROJECT_ROOT / "data" / "ambridge" / "01PassiveJud.csv")
all_verbs = list(x['Verb'].unique())
all_verbs = [x.split(" ")[0] for x in all_verbs]
inflections = [value[0] for verb in all_verbs for value in list(getAllInflectionsOOV(verb, upos='VERB').values())]
inflections = set(inflections)
with open(PROJECT_ROOT / "data" / "ambridge" / "all_verbs_conjugated.txt", 'w') as f:
    f.write('\n'.join(inflections))