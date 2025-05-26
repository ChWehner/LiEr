def reorder_line(line):
    # Split the line by tab delimiter
    elements = line.strip().split("\t")
    
    # Reorder based on new index: Index 0, Index 2, Index 1
    reordered_elements = [elements[0], elements[2], elements[1]]
    
    # Join the reordered elements back into a line with tab delimiter
    return "\t".join(reordered_elements)

def process_file(input_file, output_file):
    # Open the input file and read all lines
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # Reorder each line
    reordered_lines = [reorder_line(line) for line in lines]
    
    # Write the reordered lines to the output file
    with open(output_file, 'w') as outfile:
        outfile.write("\n".join(reordered_lines))

# Define your file paths
files = ["train.txt", "valid.txt", "test.txt"]
for file in files:
    output_file = f"reordered_{file}"
    process_file(file, output_file)
    print(f"Processed {file} -> {output_file}")
