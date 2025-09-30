import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)

input_file_path = PROJECT_ROOT / "data" / "100M" / "trigram_counts.csv"
output_file_path = PROJECT_ROOT / "data" / "100M" / "trigram_counts_formatted.csv"

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Strip whitespace and split the line by comma
        parts = line.strip().rsplit(',', 1)
        if len(parts) == 2:  # Ensure there are exactly two parts
            text, number = parts
            # Place quotation marks around the text
            modified_line = f'"{text.replace("\"", "\\\"")}",{number}\n'
            outfile.write(modified_line)
