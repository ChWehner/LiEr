# Define the path to the input and output files
input_file_path = 'entity2id.txt'
output_file_path = 'entity2id2class.txt'

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate over each line in the input file
    for line in input_file:
        # Strip newline characters and split the line into its components
        parts = line.strip().split('\t')
        # Check if the line is correctly formatted with two parts
        if len(parts) == 2:
            # Extract the identifier and the number
            identifier, number = parts
            # Identify the category based on the prefix in the identifier
            category = identifier.split('_')[1]
            # Write the modified line to the output file
            output_file.write(f"{identifier}\t{number}\t{category}\n")

print("File transformation complete. The output is saved in 'entity2id2class.txt'.")
