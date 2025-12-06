"""
Usage: python3 replace_pkt_depth.py core/src/subscription/ all
"""

import os
import sys
import re

def modify_pkt_depth(subscription_module, pkt_depth):
    for root, dirs, files in os.walk(subscription_module):
        for file in files:
            if file.endswith(".rs"):  # Modify this line to select the file types you want
                file_path = os.path.join(root, file)
                old_code = r"fn early_terminate\(&self\) -> bool \{.*?\}"
                if pkt_depth == 'all':
                    new_code = r"fn early_terminate(&self) -> bool {\n        false\n    }"
                else:
                    new_code = fr"fn early_terminate(&self) -> bool {{\n        self.cnt >= {pkt_depth}\n    }}"
                print(new_code)

                with open(file_path, 'r', encoding='utf-8') as file:
                    filedata = file.read()

                new_filedata = re.sub(old_code, new_code, filedata, flags=re.DOTALL)

                if filedata != new_filedata:
                    print(f"{file_path}: modified")
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_filedata)
                else:
                    print(f"{file_path}: no changes")

if len(sys.argv) != 3:
    print("Usage: python script.py <directory> <pkt_depth>")
    sys.exit(1)

directory = sys.argv[1]
pkt_depth = sys.argv[2]
print(directory)
print(pkt_depth)


modify_pkt_depth(directory, pkt_depth)