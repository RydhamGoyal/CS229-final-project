"""
Usage: python3 scripts/run_feature_timed.py -d /mnt/netml/results/test
"""

import subprocess, re, os
import toml
import argparse
from pprint import pprint
import hashlib
import itertools

# ANSI color codes
BLACK = '\033[30m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'
RESET = '\033[0m'
BOLD = '\033[1m'

def modify_pkt_depth(subscription_module, pkt_depth):
    old_code = r"fn early_terminate\(&self\) -> bool \{.*?\}"
    if pkt_depth == 'all':
        new_code = r"fn early_terminate(&self) -> bool {\n        false\n    }"
    else:
        new_code = fr"fn early_terminate(&self) -> bool {{\n        self.cnt >= {pkt_depth}\n    }}"
    print(new_code)
    for root, dirs, files in os.walk(subscription_module):
        for file in files:
            if file.endswith(".rs"):  # Modify this line to select the file types you want
                file_path = os.path.join(root, file)
                

                with open(file_path, 'r', encoding='utf-8') as file:
                    filedata = file.read()

                new_filedata = re.sub(old_code, new_code, filedata, flags=re.DOTALL)

                if filedata != new_filedata:
                    print(f"{file_path}: modified")
                    with open(file_path, 'w', encoding='utf-8') as file:
                        file.write(new_filedata)
                else:
                    print(f"{file_path}: no changes")


def modify_binary(template_file):
    # does nothing
    old_text = 'use retina_core::subscription::features::Features;'
    new_text = f'use retina_core::subscription::features_timed::Features;'
    with open(template_file, 'r', encoding='utf-8') as file:
        filedata = file.read()
    if old_text in filedata:
        print(GREEN + f'> Replacing `{old_text}` with `{new_text}`' + RESET)
        filedata = filedata.replace(old_text, new_text)
        with open('/home/gerryw/retina/examples/extract_features/src/main.rs', 'w', encoding='utf-8') as file:
            file.write(filedata)
        return True
    return False

def modify_config(template_file, directory):
    old_text = 'outfile = "./compute_features.csv"'
    new_text = f'outfile = "{directory}/compute_timings.csv"'
    with open(template_file, 'r', encoding='utf-8') as file:
        filedata = file.read()
    if old_text in filedata:
        print(GREEN + f'> Replacing `{old_text}` with `{new_text}`' + RESET)
        filedata = filedata.replace(old_text, new_text)
        with open('/home/gerryw/retina/scripts/tmp_config.toml', 'w', encoding='utf-8') as file:
            file.write(filedata)
        return True
    return False

def compile_binary(release):
    compile_features = f'timing'
    status = True
    if release:
        cmd = f'cargo build --release --bin extract_features --features {compile_features}'
    else:
        cmd = f'cargo build --bin extract_features --features {compile_features}'
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    for stderr_line in iter(popen.stderr.readline, ''):
        # print(stderr_line, end='')
        if 'Compiling' in stderr_line or 'Finished' in stderr_line:
            print(f'\t> {stderr_line}', end='')
        if 'error' in stderr_line:
            print(RED + f'\t> {stderr_line}', end='' + RESET)
            status = False
    popen.stdout.close()
    popen.stderr.close()
    popen.wait()
    return status

def run_binary(directory, release, online):
    status = True
    binary = 'extract_features'
    if release:
        executable = f'/home/gerryw/retina/target/release/{binary}'
    else:
        executable = f'/home/gerryw/retina/target/debug/{binary}'
    
    config_file = '/home/gerryw/retina/scripts/tmp_config.toml'
    out_file = f'{directory}/out_features.json'
    cmd = f'sudo env LD_LIBRARY_PATH=$LD_LIBRARY_PATH RUST_LOG=info {executable} -c {config_file} -o {out_file}'

    print(GREEN + f'> Running `{cmd}`' + RESET)

    EPSILON = 0.0001

    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ''):
        print(stdout_line, end='')
        if 'SW Dropped' in stdout_line:
            num = re.findall('\d*\.*\d+\%', stdout_line)
            if not num: continue
            value = float(num[0].split('%')[0]) 
            if value > EPSILON :
                print(RED + f'> TERMINATING, current SW drops {value} greater than {EPSILON}...' + RESET)
                stream = os.popen(f'pidof {binary}')
                pid = stream.read()
                os.system(f'sudo kill -INT {pid}')
                status = False     # Failed, skip
    popen.stdout.close()
    popen.wait()
    return status

def get_feature_subsets(ft_names, r=None):
    if r:
        return list(itertools.combinations(ft_names, r))
    subsets = []
    n = len(ft_names)
    for i in range(1, 2**n):
        subset = []
        for j in range(n):
            if i & (1 << j):
                subset.append(ft_names[j])
        subsets.append(sorted(subset))
    return subsets

def main(args):

    pkt_depths = ['all',1,2,3,4,5,6,7,8,9,10,100,1000,10000]

    errors = {}
    
    for pkt_depth in pkt_depths:
        print(YELLOW +BOLD + str(pkt_depth) + RESET)
        modify_pkt_depth('/home/gerryw/retina/core/src/subscription/', pkt_depth)
        directory = f'{args.directory}/pkts_{pkt_depth}'
        os.makedirs(directory, exist_ok=True)
        print(YELLOW + directory + RESET)

        if args.online:
            config_file = '/home/gerryw/retina/scripts/base_online_config.toml'
        else:
            config_file = '/home/gerryw/retina/scripts/base_offline_config.toml'

        if not modify_binary('/home/gerryw/retina/scripts/base_extract_features.rs'):
            print(f'Failed to modify binary template, skipping...')
            errors[pkt_depth] = 'apply_binary_template'
        
        if not modify_config(config_file, directory):
            print(f'Failed to modify config template, skipping...')
            errors[pkt_depth] = 'modify_config'
        
        if not compile_binary(release=args.release):
            print(f'Failed to compile, skipping...')
            errors[pkt_depth] = 'compile'
            continue
    
        if not run_binary(directory, release=args.release, online=args.online):
            print(f'Failed to run, skipping...')
            errors[pkt_depth] = 'runtime'
            continue


    for error in errors.items():
        print(RED + f'Error: {str(error)}' + RESET)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory')
    parser.add_argument('--release', action='store_true', default=False)
    parser.add_argument('--online', action='store_true', default=False)
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
