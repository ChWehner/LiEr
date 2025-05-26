def switch_positions(input_file_path, output_file_path):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        for line in input_file:
            parts = line.strip().split('\t')  # Assuming the separator is a tab character
            if len(parts) == 2:  # To ensure there are exactly two elements to swap
                # Swapping the position of ID and entity/relation and writing to the output file
                output_file.write(f"{parts[1]}\t{parts[0]}\n")

# You can call the function for each of your files like so:
switch_positions('entity2id.txt', 'id2entity.txt')
switch_positions('relation2id.txt', 'id2relation.txt')

