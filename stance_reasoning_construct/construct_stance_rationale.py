import sys
sys.path.append('.')
import os
from stance_reasoning_construct.openai_api_caller import openai_api_caller
from prompts import generate_few_shot_prompt
from utils.dataset_utils.data_config import data_configs
from utils.tools import load_csv, save_csv
from dotenv import load_dotenv
import logging

logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(filename)-20s : %(lineno)s line - %(message)s'
DATE_FORMAT = '%Y-%m-%d_%H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)

import sys
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = handle_exception
load_dotenv()
regex_pattern = r'"answer":\s*["\']?(.*?)["\']?\s*[,]?\s*"stance":\s*["\']?(favor|against|neutral)["\']?'

def main(llm_name):

    if llm_name == 'gpt':
        service_url = os.getenv("OPENAI_URL")
        model_name = 'gpt-3.5-turbo-0125'
        api_key = os.getenv("OPENAI_API_KEY")
    elif llm_name == 'llama':
        service_url = os.getenv("LLAMA2_URL")
        model_name = 'Llama-2-70b-chat-hf'
        api_key = os.getenv("LLAMA2_API_KEY")

    for dataset_name, dataset_config in data_configs.items():

        llm_responses = {}
        if dataset_name != 'vast':
            for target_name in dataset_config.zero_shot_target_names:
                all_datas = load_csv(f'{dataset_config.zero_shot_data_dir}/{target_name}/test.csv')
                prompts = [generate_few_shot_prompt(dataset_name, data['Tweet'], data['Target']) for data in all_datas]
                responses = openai_api_caller(prompts, model_name, api_key, regex_pattern=regex_pattern, service_url=service_url, parallel_num=15)
                for data, response in zip(all_datas, responses):
                    llm_responses[f"{data['Target']}:{data['Tweet']}"] = response
        else:
            all_datas = load_csv(f'{dataset_config.zero_shot_data_dir}/zero-shot/train.csv')
            all_datas += load_csv(f'{dataset_config.zero_shot_data_dir}/zero-shot/valid.csv')
            all_datas += load_csv(f'{dataset_config.zero_shot_data_dir}/zero-shot/test.csv')
            prompts = [generate_few_shot_prompt(dataset_name, data['Tweet'], data['Target']) for data in all_datas]
            responses = openai_api_caller(prompts, model_name, api_key, regex_pattern=regex_pattern, service_url=service_url, parallel_num=15)
            for data, response in zip(all_datas, responses):
                llm_responses[f"{data['Target']}:{data['Tweet']}"] = response

        if dataset_name != 'vast':
            # in-target
            for target_name in dataset_config.in_target_target_names:
                for data_split in ['train', 'valid', 'test']:
                    all_datas = load_csv(f'{dataset_config.in_target_data_dir}/{target_name}/{data_split}.csv')
                    for data in all_datas:
                        if len(llm_responses[f"{data['Target']}:{data['Tweet']}"]) == 1:
                            data['llm_stance'] = None
                            data['llm_rationale'] = None
                        else:
                            data['llm_stance'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][1]
                            data['llm_rationale'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][0]
                    write_path = f"{dataset_config.in_target_llm_rationale_data_dir[llm_name]}/{target_name}/{data_split}.csv"
                    if not os.path.isdir(os.path.dirname(write_path)):
                        os.makedirs(os.path.dirname(write_path))
                    save_csv(all_datas, write_path)
            # zero-shot
            for target_name in dataset_config.zero_shot_target_names:
                for data_split in ['train', 'valid', 'test']:
                    all_datas = load_csv(f'{dataset_config.zero_shot_data_dir}/{target_name}/{data_split}.csv')
                    for data in all_datas:
                        if len(llm_responses[f"{data['Target']}:{data['Tweet']}"]) == 1:
                            data['llm_stance'] = None
                            data['llm_rationale'] = None
                        else:
                            data['llm_stance'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][1]
                            data['llm_rationale'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][0]
                    write_path = f"{dataset_config.zero_shot_llm_rationale_data_dir[llm_name]}/{target_name}/{data_split}.csv"
                    if not os.path.isdir(os.path.dirname(write_path)):
                        os.makedirs(os.path.dirname(write_path))
                    save_csv(all_datas, write_path)
        else:
            # zero-shot
            for data_split in ['train', 'valid', 'test']:
                all_datas = load_csv(f'{dataset_config.zero_shot_data_dir}/zero-shot/{data_split}.csv')
                for data in all_datas:
                    if len(llm_responses[f"{data['Target']}:{data['Tweet']}"]) == 1:
                        data['llm_stance'] = None
                        data['llm_rationale'] = None
                    else:
                        data['llm_stance'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][1]
                        data['llm_rationale'] = llm_responses[f"{data['Target']}:{data['Tweet']}"][0]
                write_path = f"{dataset_config.zero_shot_llm_rationale_data_dir[llm_name]}/zero-shot/{data_split}.csv"
                if not os.path.isdir(os.path.dirname(write_path)):
                    os.makedirs(os.path.dirname(write_path))
                save_csv(all_datas, write_path)


if __name__ == "__main__":
    # gpt, llama
    main('gpt')
    main('llama')