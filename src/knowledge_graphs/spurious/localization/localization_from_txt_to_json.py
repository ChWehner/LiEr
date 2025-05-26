#!/usr/bin/env python3
import json
import re
import os
import sys
import glob


def extract_json_objects(text):
    """
    Extracts all JSON objects from the input text using a non-greedy regular expression.
    """
    pattern = re.compile(r"\{.*?\}", re.DOTALL)
    return pattern.findall(text)


def transform_file(input_file):
    """
    Reads the input file, extracts JSON objects, and transforms them into a single dictionary.
    """
    with open(input_file, "r") as file:
        text = file.read()

    json_strs = extract_json_objects(text)
    transformed = {}
    for index, js in enumerate(json_strs):
        try:
            obj = json.loads(js)
            transformed[str(index)] = obj
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {input_file} at index {index}: {e}")
    return transformed


def process_folder(folder):
    """
    Processes all .txt files in the given folder, transforms their content,
    and saves the output with a .json extension.
    """
    txt_files = glob.glob(os.path.join(folder, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {folder}.")
        return

    for txt_file in txt_files:
        transformed = transform_file(txt_file)
        base_name = os.path.splitext(txt_file)[0]
        output_file = base_name + ".json"
        with open(output_file, "w") as outfile:
            json.dump(transformed, outfile, indent=2)
        print(f"Processed {txt_file} -> {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_folder.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: {folder} is not a valid directory.")
        sys.exit(1)

    process_folder(folder)


if __name__ == "__main__":
    main()
