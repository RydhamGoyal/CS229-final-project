"""
Usage: python3 scripts/run_feature_collect.py -d /mnt/netml/datasets/test --release
"""

import subprocess, re, os
import toml
import argparse
from pprint import pprint

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



def compile_binary(feature_comma, release):
    compile_features = feature_comma
    status = True
    if release:
        cmd = f'cargo build --release --bin log_features --features {compile_features}'
    else:
        cmd = f'cargo build --bin log_features --features {compile_features}'
    print(CYAN + cmd + RESET)
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

def modify_config(template_file):
    # make no modifications, just use the base config whether online or offline
    with open(template_file, 'r', encoding='utf-8') as file:
        filedata = file.read()
    
    with open('/home/gerryw/retina/scripts/tmp_config.toml', 'w', encoding='utf-8') as file:
        file.write(filedata)
        return True

def run_binary(directory, release):
    status = True
    binary = 'log_features'
    if release:
        executable = f'/home/gerryw/retina/target/release/{binary}'
    else:
        executable = f'/home/gerryw/retina/target/debug/{binary}'
    
    config_file = '/home/gerryw/retina/scripts/tmp_config.toml'

    out_file = f'{directory}/features.jsonl'
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


def main(args):

    ft_names = [
        'dur',
        'proto',
        's_bytes_sum',
        'd_bytes_sum',
        's_ttl_mean',
        'd_ttl_mean',
        's_load',
        'd_load',
        's_bytes_mean',
        'd_bytes_mean',
        's_pkt_cnt',
        'd_pkt_cnt',
        's_iat_mean',
        'd_iat_mean',
        'tcp_rtt',
        'syn_ack',
        'ack_dat',
    ]

    pkt_depths = [1,2,3,4,5,6,7,8,9,10,20,50,100,1000,10000,'all']

    errors = {}
    for pkt_depth in pkt_depths:
        print(YELLOW +BOLD + str(pkt_depth) + RESET)
        modify_pkt_depth('/home/gerryw/retina/core/src/subscription/', pkt_depth)
        directory = f'{args.directory}/pkts_{pkt_depth}'
        os.makedirs(directory, exist_ok=True)
        print(YELLOW +BOLD + directory + RESET)

        if args.online:
            config_file = '/home/gerryw/retina/scripts/base_online_config.toml'
        else:
            config_file = '/home/gerryw/retina/scripts/base_offline_config.toml'
        
        if not modify_config(config_file):
            print(f'Failed to modify config template for `{pkt_depth}`, skipping...')
            errors[pkt_depth] = 'modify_config'
        
        feature_comma = ','.join(sorted(ft_names))
        if not compile_binary(feature_comma, release=args.release):
            print(f'Failed to compile for `{pkt_depth}`, skipping...')
            errors[pkt_depth] = 'compile'
            continue

        if not run_binary(directory, release=args.release):
            print(f'Failed to run {pkt_depth}, skipping...')
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
