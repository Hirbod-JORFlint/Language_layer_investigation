def extract_labels_from_ud_file(path):
    labels = set()
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line == '\n' or line[0] == '#':
                continue
            line = line.rstrip('\n')
            line = line.split('\t')
            if len(line) > 7:  # Ensure the line has a dependency label
                labels.add(line[7])
    return labels

# Usage:
path_to_ud_file = 'E:\\Dars\\Research Project\\Language_layer_investigation\\en_ewt-ud-train.conllu'  # Replace with your file path
unique_labels = extract_labels_from_ud_file(path_to_ud_file)
print(f"Unique labels in the file: {unique_labels}")
