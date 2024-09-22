import os
import csv
import math
import json
import shutil
import natsort
import hashlib

def load_dict(data_path):
    with open(data_path, 'r', encoding='utf-8') as read_f:
        data = json.load(read_f)
    return data

def save_dict(data, data_path):
    with open(data_path, 'w', encoding='utf-8') as write_f:
        json.dump(data, write_f)

def load_csv(data_path):
    all_datas = []
    with open(data_path, 'r', encoding='utf-8') as read_f:
        reader = csv.DictReader(read_f)
        for row in reader:
            all_datas.append(row)
    return all_datas

def save_csv(datas, data_path, mode='w'):
    if len(datas) == 0:
        return
    field_names = datas[0].keys()
    if not os.path.exists(data_path) or mode == 'w':
        with open(data_path, 'w', encoding='utf-8') as write_f:
            writer = csv.DictWriter(write_f, fieldnames=field_names)
            writer.writeheader()

    with open(data_path, 'a', encoding='utf-8') as write_f:
        writer = csv.DictWriter(write_f, fieldnames=field_names)
        for data in datas:
            writer.writerow(data)

def create_csv_spilts(prompts, system_prompts, output_dir, cut_nums):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # The maximum number of splits is the total number of data
    cut_nums = min(len(prompts), cut_nums)
    num_per_file = math.ceil(len(prompts) / cut_nums)

    for i in range(cut_nums):
        write_datas = []
        save_path = f'{output_dir}/{i}.csv'
        if system_prompts == None:
            for prompt in prompts[i*num_per_file:(i+1)*num_per_file]:
                write_data = {
                    'prompt': prompt,
                }
                write_datas.append(write_data)
        else:
            for prompt, system_prompt in zip(prompts[i*num_per_file:(i+1)*num_per_file], system_prompts[i*num_per_file:(i+1)*num_per_file]):
                write_data = {
                    'prompt': prompt,
                    'system_prompt': system_prompt,
                }
                write_datas.append(write_data)

        save_csv(write_datas, save_path)
    save_cache_identify(prompts, system_prompts, output_dir)

def merge_csv(cache_data_paths, output_path):
    if len(cache_data_paths) == 0:
        return
    response_datas = []
    save_datas = []
    for d in cache_data_paths:
        save_datas += load_csv(d)
    if output_path is not None:
        save_csv(save_datas, output_path)
    for data in save_datas:
        response_datas.append(eval(data['llm_response']))
    shutil.rmtree(os.path.dirname(cache_data_paths[0]))
    return response_datas

def save_cache_identify(prompts, system_prompts, cache_file_dir):
    prompts_hash = hashlib.sha256(str(prompts).encode('utf-8')).hexdigest()
    if system_prompts == None:
        system_prompts_hash = hashlib.sha256(str(None).encode('utf-8')).hexdigest()
    else:
        system_prompts_hash = hashlib.sha256(str(system_prompts).encode('utf-8')).hexdigest()
    
    cache_identify = {
        'prompts_hash': prompts_hash,
        'system_prompts_hash': system_prompts_hash,
    }
    save_dict(cache_identify, f'{cache_file_dir}/cache_identify.json')

def check_cache_identify(prompts, system_prompts, cache_file_dir):
    prompts_hash = hashlib.sha256(str(prompts).encode('utf-8')).hexdigest()
    if system_prompts == None:
        system_prompts_hash = hashlib.sha256(str(None).encode('utf-8')).hexdigest()
    else:
        system_prompts_hash = hashlib.sha256(str(system_prompts).encode('utf-8')).hexdigest()
    
    if not os.path.exists(f'{cache_file_dir}/cache_identify.json'):
        return False
    cache_identify = load_dict(f'{cache_file_dir}/cache_identify.json')
    if cache_identify['prompts_hash'] == prompts_hash and cache_identify['system_prompts_hash'] == system_prompts_hash:
        return True
    else:
        return False

def get_cache_data_paths(cache_file_path):
    cache_data_paths = [path for path in os.listdir(cache_file_path) if (path.endswith('.csv') and not path.endswith('.tmp.csv'))]
    cache_data_paths = natsort.natsorted(cache_data_paths)
    cache_data_paths = [os.path.join(cache_file_path, d) for d in cache_data_paths]
    return cache_data_paths