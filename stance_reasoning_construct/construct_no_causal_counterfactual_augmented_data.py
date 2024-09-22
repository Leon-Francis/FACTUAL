import sys
sys.path.append('.')
import os
from stance_reasoning_construct.openai_api_caller import openai_api_caller
from prompts import CONSTRUCT_NON_CAUSAL_COUNTERFACTUAL_PROMPT as PROMPTS
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
regex_pattern = r'"Revised Sentence":\s*["\']?(.*?)["\']?\s*[,]?\s*"Rephrased Topic":\s*["\']?(.*?)["\']?'

DATASET_DIR = 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_non_causal_counterfactural_dataset'
DATASET_FILE_PATHS = {
    'sem16': 'Semeval16/semeval16.csv',
    'p_stance': 'P-Stance/p-stance.csv',
    'vast': 'VAST/vast_train.csv'
}

p_stance_map_label = {'FAVOR': 'support', 'AGAINST': 'against'}
vast_map_label = {'pro': 'support', 'con': 'against', 'neutral': 'unrelated'}
covid_19_map_label = {'FAVOR': 'support', 'AGAINST': 'against', 'NONE': 'neutral'}
semeval16_map_label = {'FAVOR': 'support', 'AGAINST': 'against', 'NONE': 'neutral'}

def main():
    service_url = os.getenv("OPENAI_URL")
    model_name = 'gpt-3.5-turbo-0301'
    api_key = os.getenv("OPENAI_API_KEY")

    for dataset_name, dataset_config in data_configs.items():
        all_datas = []
        if dataset_name != 'vast':
            for target_name in dataset_config.zero_shot_target_names:
                all_datas += load_csv(f'{dataset_config.zero_shot_data_dir}/{target_name}/test.csv')
        else:
            all_datas += load_csv(f'{dataset_config.zero_shot_data_dir}/zero-shot/train.csv')
        
        prompts = []
        dataset_map_label = eval(f'{dataset_name}_map_label')
        for data in all_datas:
            label = dataset_map_label[data['label']]
            if label == 'unrelated':
                prompts.append(PROMPTS['unrelated_prompt'] % (data['tweet_text'], data['target']))
            else:
                prompts.append(PROMPTS['stance_prompt'] % (label, label, data['tweet_text'], data['target']))
        responses = openai_api_caller(prompts, model_name, api_key, system_prompts=PROMPTS['system_prompt'], regex_pattern=regex_pattern, service_url=service_url, parallel_num=15)
        for data, response in zip(all_datas, responses):
            data['non_causal_counterfactural_text'] = response[0]
            data['non_causal_counterfactural_target'] = response[1]
        save_csv(all_datas, f'{DATASET_DIR}/{DATASET_FILE_PATHS[dataset_name]}')


if __name__ == '__main__':
    main()