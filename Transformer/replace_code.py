"""
Usage: python3 replace_code.py core/src/subscription/
"""

import os
import sys

def replace_in_file(file_path, old_text, new_text):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        filedata = file.read()

    if old_text in filedata:
        print(f"{file_path}: modified")
    else:
        print(f"{file_path}: no changes")
    # Replace the target string
    filedata = filedata.replace(old_text, new_text)
    # Write the file out again
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(filedata)

old_block = """fn early_terminate(&self) -> bool {
        self.cnt >= 1
    }"""
new_block = """fn early_terminate(&self) -> bool {
        false
    }"""

if len(sys.argv) != 2:
    print("Usage: python script.py <directory>")
    sys.exit(1)

directory = sys.argv[1]
print(directory)

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith(".rs"):  # Modify this line to select the file types you want
            file_path = os.path.join(root, file)
            replace_in_file(file_path, old_block, new_block)
