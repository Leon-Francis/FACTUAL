from typing import List, Tuple, Optional
import multiprocessing
from func_timeout import func_timeout, FunctionTimedOut
from functools import partial
import re
import os
import time
from tqdm import tqdm
from datetime import datetime
import requests
import logging
from stance_reasoning_construct.openai_api_caller_utils import load_csv, save_csv, check_cache_identify, create_csv_spilts, get_cache_data_paths, merge_csv
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


def get_response(
    prompt: str, 
    model_name: str,
    api_key: str,
    system_prompt: Optional[str] = None, 
    regex_pattern: Optional[str] = None, 
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seeds: List[int] = [42, 43, 44, 45, 46],
    service_url: str = 'https://api.openai.com/v1/chat/completions',
    seed_idx: int = 0,
    service_error_try: int = 0, 
    parse_error_try: int = 0,
) -> Tuple[str]:
    """
    Get the response from the OpenAI API based on the given prompt.

    Args:
        prompt (str): The user's prompt.
        model_name (str): The name of the model to use.
        system_prompt (Optional[str], optional): The system prompt. Defaults to None.
        regex_pattern (Optional[str], optional): The regex pattern to match in the response. Defaults to None.
        temperature (float, optional): The temperature parameter for generating the response. Defaults to 1.0.
        top_p (float, optional): The top-p parameter for generating the response. Defaults to 1.0.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 1024.
        seeds (List[int], optional): The list of seed values. Defaults to [42, 43, 44, 45, 46].
        seed_idx (int, optional): The index of the seed value to use. Defaults to 0.
        service_url (str, optional): URL of the OpenAI API service. Defaults to 'https://api.openai.com/v1/chat/completions'.
        service_error_try (int, optional): The number of times to retry in case of service errors. Defaults to 0.
        parse_error_try (int, optional): The number of times to retry in case of parsing errors. Defaults to 0.

    Returns:
        Tuple[str]: A tuple containing the response from the OpenAI API.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if len(enc.encode(prompt)) > 3800:
        prompt = enc.decode(enc.encode(prompt)[:3800])
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ] if system_prompt != None else [{"role": "user", "content": prompt}],
        "seed": seeds[seed_idx],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    try:
        response = ''
        response = requests.post(service_url, headers=headers, json=payload).json()
        if 'system_fingerprint' in response:
            logging.info(f'Model {model_name} request finish, system fingerprint: {response["system_fingerprint"]}')
        else:
            logging.info(f'Model {model_name} request finish, no system fingerprint in response')
        response = response['choices'][0]['message']['content'].strip()
        if len(re.findall(regex_pattern, response)) == 0:
            raise ValueError("parse pattern not detected.")
        elif len(re.findall(regex_pattern, response)) > 1:
            raise ValueError("Detect more than on matched result.")
        if type(re.findall(regex_pattern, response)[0]) == tuple:
            return re.findall(regex_pattern, response)[0]
        else:
            return (re.findall(regex_pattern, response)[0], )
    except (KeyError, requests.exceptions.ProxyError, requests.exceptions.SSLError, requests.exceptions.ConnectionError, ConnectionRefusedError) as e:
        # Service Error
        logging.error(f'Get Service Error {e.__class__.__name__}: {e}')
        logging.error(f'The trigger text is {response}')
        time.sleep(2**service_error_try)
        return get_response(prompt, model_name, api_key, system_prompt, regex_pattern, temperature, top_p, max_tokens, seeds, service_url, seed_idx, service_error_try, parse_error_try+1)
    except (ValueError, IndexError) as e:
        # Parsing Error
        logging.error(f'Get Parsing Error {e.__class__.__name__}: {e}')
        logging.error(f'The trigger text is {response}')
        if parse_error_try > 4:
            return (None, )
        time.sleep(2**service_error_try)
        return get_response(prompt, model_name, api_key, system_prompt, regex_pattern, temperature, top_p, max_tokens, seeds, service_url, seed_idx, service_error_try, parse_error_try+1)

def multiprocess_llm_infer(read_data_path, model_name, api_key, regex_pattern, temperature, top_p, max_tokens, seeds, service_url):
    datas = load_csv(read_data_path)
    write_temp_path = os.path.splitext(read_data_path)[0] + '.tmp.csv'

    # Use the cached results.
    if os.path.exists(write_temp_path):
        write_temp_datas = load_csv(write_temp_path)
        datas = datas[len(write_temp_datas):]

    for data in tqdm(datas):
        system_prompt = data['system_prompt'] if 'system_prompt' in data else None
        prompt = data['prompt']
        llm_response = get_response(
            prompt,
            model_name,
            api_key,
            system_prompt,
            regex_pattern,
            temperature,
            top_p,
            max_tokens,
            seeds,
            service_url
        )
        data['llm_response'] = llm_response
        save_csv([data], write_temp_path, mode='a')

def load_cache_file_with_timeout(cache_files, timeout=60):
    logging.info(f'*{len(cache_files)} usable cache files detected!*')
    for idx, dir_name in enumerate(cache_files):
        print(f'{idx}: {dir_name}')
    print(f'Which cache do you want to read? Input the number, N for Do not read cache:')

    try:
        result = func_timeout(timeout, lambda: input())
    except FunctionTimedOut:
        print(f'*Input timeout, cache not read by default.*')
        return None

    if result is None or result == 'N' or result == 'n':
        return None
    else:
        if result.isdigit() and int(result) < len(cache_files):
            return cache_files[int(result)]
        else:
            return load_cache_file_with_timeout(cache_files, timeout)

def check_cache_files_usable(prompts, system_prompts, cache_files_dir):
    cache_files = []
    for dir_name in [d for d in os.listdir(cache_files_dir)]:
        if check_cache_identify(prompts, system_prompts, f'{cache_files_dir}/{dir_name}'):
            cache_files.append(dir_name)

    if len(cache_files) == 0:
        return None
    return cache_files

def openai_api_caller(
    prompts: List[str],
    model_name: str,
    api_key: str,
    system_prompts = None,
    saved_path: Optional[str] = None,
    regex_pattern: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    seeds: List[int] = [42, 43, 44, 45, 46],
    parallel_num: int = 10,
    cache_files_dir: str = '.cache',
    service_url: str = 'https://api.openai.com/v1/chat/completions'
) -> List[Tuple[str]]:

    """
    Calls the OpenAI API to generate responses for the given prompts using the specified model.

    Args:
        prompts (List[str]): List of prompts for which responses need to be generated.
        model_name (str): Name of the OpenAI model to be used for generating responses.
        system_prompts (Optional[List[str]|str], optional): List of system prompts to be used for generating responses. Defaults to None.
        saved_path (Optional[str], optional): Path to save the generated responses. Defaults to None.
        regex_pattern (Optional[str], optional): Regular expression pattern to filter the generated responses. Defaults to None.
        temperature (float, optional): Controls the randomness of the generated responses. Higher values make the responses more random. Defaults to 1.0.
        top_p (float, optional): Controls the diversity of the generated responses. Lower values make the responses more focused. Defaults to 1.0.
        max_tokens (int, optional): Maximum number of tokens in the generated responses. Defaults to 1024.
        seeds (List[int], optional): List of random seeds for generating responses. Defaults to [42, 43, 44, 45, 46].
        parallel_num (int, optional): Number of parallel processes for generating responses. Defaults to 10.
        cache_files_dir (str, optional): Directory to store cache files. Defaults to '.cache'.
        service_url (str, optional): URL of the OpenAI API service. Defaults to 'https://api.openai.com/v1/chat/completions'.

    Returns:
        List[Tuple[str]]: List of tuples containing the generated responses for each prompt.
    """

    now_time = datetime.now().strftime("%Y-%m-%d %H_%M_%S")

    if not os.path.exists(cache_files_dir):
        os.makedirs(cache_files_dir)

    if type(system_prompts) == str:
        system_prompts = [system_prompts] * len(prompts)

    cache_files = check_cache_files_usable(prompts, system_prompts, cache_files_dir)

    if cache_files != None:
        cache_file = load_cache_file_with_timeout(cache_files)
        if cache_file != None:
            cache_file_path = f'{cache_files_dir}/{cache_file}'
            logging.info(f'*Using cache file in {cache_file_path}*')
        else:
            cache_file_path = f'{cache_files_dir}/{now_time}'
            logging.info(f'*Generate cache file in {cache_file_path}*')
            create_csv_spilts(prompts, system_prompts, cache_file_path, parallel_num)
    else:
        cache_file_path = f'{cache_files_dir}/{now_time}'
        logging.info(f'*Generate cache file in {cache_file_path}*')
        create_csv_spilts(prompts, system_prompts, cache_file_path, parallel_num)

    cache_data_paths = get_cache_data_paths(cache_file_path)

    # multiprocess llm infer
    logging.info(f'*Inferencing, parallel nums: {parallel_num}*')

    pool = multiprocessing.Pool(processes=parallel_num)
    func = partial(
        multiprocess_llm_infer, 
        model_name=model_name,
        api_key=api_key,
        regex_pattern=regex_pattern,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seeds=seeds,
        service_url=service_url
    )
    pool.map(func, cache_data_paths)
    pool.close()
    pool.join()

    # merge csv
    logging.info('*Merging results*')
    cache_response_data_paths = [os.path.splitext(data_path)[0] + '.tmp.csv' for data_path in cache_data_paths]
    response_datas = merge_csv(cache_response_data_paths, saved_path)

    return response_datas