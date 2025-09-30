import pandas as pd
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

# Read in trigram_counts.csv as a dictionary
trigram_counts = pd.read_csv(PROJECT_ROOT / "data" / "100M" / "trigram_counts_formatted.csv")