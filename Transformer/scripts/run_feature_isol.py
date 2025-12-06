"""
Usage: python3 scripts/run_feature_isol.py -d /mnt/netml/results/test
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


def modify_binary(template_file, feature):
    old_text = 'use retina_core::subscription::features::Features;'
    new_text = f'use retina_core::subscription::features_{feature}::Features;'
    with open(template_file, 'r', encoding='utf-8') as file:
        filedata = file.read()
    if old_text in filedata:
        print(GREEN + f'> Replacing `{old_text}` with `{new_text}`' + RESET)
        filedata = filedata.replace(old_text, new_text)
        with open('/home/gerryw/retina/examples/extract_features/src/main.rs', 'w', encoding='utf-8') as file:
            file.write(filedata)
        return True
    return False

def modify_config(template_file, directory, feature):
    old_text = 'outfile = "./compute_features.csv"'
    new_text = f'outfile = "{directory}/compute_features_{feature}.csv"'
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
    status = True
    if release:
        cmd = f'cargo build --release --bin extract_features --features timing'
    else:
        cmd = f'cargo build --bin extract_features --features timing'
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

def run_binary(directory, feature, release, online):
    status = True
    binary = 'extract_features'
    if release:
        executable = f'/home/gerryw/retina/target/release/{binary}'
    else:
        executable = f'/home/gerryw/retina/target/debug/{binary}'
    
    config_file = '/home/gerryw/retina/scripts/tmp_config.toml'
    out_file = f'{directory}/out_features_{feature}.json'
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

def profile_memory(directory, feature):
    out_file = f'{directory}/mem_features_{feature}.txt'
    # cmd = f'top -d 1 -b | grep --line-buffered extract_feature'
    cmd = f'top -d 1 -b | grep --line-buffered extract_feature > {out_file}'

    print(GREEN + f'> Starting memory profiler, writing to `{out_file}`' + RESET)
    popen = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, universal_newlines=True)
    return popen

def main(args):
    ft_names = [
        'all',
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

    pkt_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'all']

    errors = {}
    for pkt_depth in pkt_depths:
        print(YELLOW +BOLD + str(pkt_depth) + RESET)
        modify_pkt_depth('/home/gerryw/retina/core/src/subscription/', pkt_depth)
        directory = f'{args.directory}/pkts_{pkt_depth}'
        os.makedirs(directory, exist_ok=True)
        print(YELLOW +BOLD + directory + RESET)
        
        for feature in ft_names:
            print(CYAN + BOLD + feature + RESET)
            
            if not modify_binary('/home/gerryw/retina/scripts/base_extract_features.rs', feature):
                print(f'Failed to modify binary template for `{feature}`, skipping...')
                errors[feature] = 'modify_binary'
            
            if args.online:
                config_file = '/home/gerryw/retina/scripts/base_online_config.toml'
            else:
                config_file = '/home/gerryw/retina/scripts/base_offline_config.toml'
            
            if not modify_config(config_file, directory, feature):
                print(f'Failed to modify config template for `{feature}`, skipping...')
                errors[feature] = 'modify_config'
            
            if not compile_binary(release=args.release):
                print(f'Failed to compile for `{feature}`, skipping...')
                errors[feature] = 'compile'
                continue
        
            mem_profiler = profile_memory(directory, feature)
            if not run_binary(directory, feature, release=args.release, online=args.online):
                print(f'Failed to run {feature}, skipping...')
                errors[feature] = 'runtime'
                continue
            mem_profiler.terminate()
            # terminate won't kill the child top process since it is running with shell=True
            os.system(f'pkill -f top')

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
