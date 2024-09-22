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

def main(llm_name, causal):

    if llm_name == 'gpt':
        service_url = os.getenv("OPENAI_URL")
        model_name = 'gpt-3.5-turbo-0125'
        api_key = os.getenv("OPENAI_API_KEY")
    elif llm_name == 'llama':
        service_url = os.getenv("LLAMA2_URL")
        model_name = 'Llama-2-70b-chat-hf'
        api_key = os.getenv("LLAMA2_API_KEY")

    causal_data_paths = {
        'sem16': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_causal_counterfactural_dataset/Semeval16/semeval16.csv',
        'p_stance': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_causal_counterfactural_dataset/P-Stance/p-stance.csv',
        'vast': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_causal_counterfactural_dataset/VAST/vast_train.csv'
    }
    non_causal_data_paths = {
        'sem16': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/semeval16.csv',
        'p_stance': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/p-stance.csv',
        'vast': 'datasets/conterfactual_datasets/conterfactual_generate_datasets/gpt_non_causal_counterfactural_dataset/VAST/vast_train.csv'
    }

    if causal == 'causal':
        data_paths = causal_data_paths
    elif causal == 'non_causal':
        data_paths = non_causal_data_paths
    else:
        raise ValueError

    for dataset_name, dataset_config in data_configs.items():

        all_counterfactual_datas = load_csv(data_paths[dataset_name])
        if causal == 'causal':
            prompts = [generate_few_shot_prompt(dataset_name, data['causal_sentence_counterfactural'], data['target']) for data in all_counterfactual_datas]
        elif causal == 'non_causal':
            prompts = [generate_few_shot_prompt(dataset_name, data['non_causal_counterfactural_text'], data['non_causal_counterfactural_target']) for data in all_counterfactual_datas]
        else:
            raise ValueError
        responses = openai_api_caller(prompts, model_name, api_key, regex_pattern=regex_pattern, service_url=service_url, parallel_num=15)
        for data, response in zip(all_counterfactual_datas, responses):
            if len(response) == 1:
                data['llm_stance'] = None
                data['llm_rationale'] = None
            else:
                data['llm_stance'] = response[1]
                data['llm_rationale'] = response[0]
        write_path = f"{data_paths[dataset_name].replace('conterfactual_generate_datasets', f'conterfactual_generate_datasets/{llm_name}_analysis_datasets')}"
        if not os.path.isdir(os.path.dirname(write_path)):
            os.makedirs(os.path.dirname(write_path))
        save_csv(all_counterfactual_datas, write_path)

def process_counterfactual_augmentation_data(llm_name):
    causal_data_paths = {
        'sem16': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_causal_counterfactural_dataset/Semeval16/semeval16.csv',
        'p_stance': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_causal_counterfactural_dataset/P-Stance/p-stance.csv',
        'vast': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_causal_counterfactural_dataset/VAST/vast_train.csv'
    }
    non_causal_data_paths = {
        'sem16': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_non_causal_counterfactural_dataset/Semeval16/semeval16.csv',
        'p_stance': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_non_causal_counterfactural_dataset/P-Stance/p-stance.csv',
        'vast': f'datasets/conterfactual_datasets/conterfactual_generate_datasets/{llm_name}_analysis_datasets/gpt_non_causal_counterfactural_dataset/VAST/vast_train.csv'
    }
    for dataset_name, dataset_config in data_configs.items():
        error_num = 0
        causal_datas = load_csv(causal_data_paths[dataset_name])
        non_causal_datas = load_csv(non_causal_data_paths[dataset_name])
        causal_data2idx = {f"{data['tweet_text']}:{data['target']}": idx for idx, data in enumerate(causal_datas)}
        non_causal_data2idx = {f"{data['tweet_text']}:{data['target']}": idx for idx, data in enumerate(non_causal_datas)}
        # in-target
        if dataset_name != 'vast':
            for target_name in dataset_config.in_target_target_names:
                all_datas = load_csv(f'{dataset_config.in_target_data_dir}/{target_name}/train.csv')
                write_causal_all_datas = []
                write_non_causal_all_datas = []
                for data in all_datas:
                    if f"{data['Tweet']}:{data['Target']}" in causal_data2idx:
                        causal_data = causal_datas[causal_data2idx[f"{data['Tweet']}:{data['Target']}"]]
                        new_data = {}
                        new_data['Tweet'] = causal_data['causal_sentence_counterfactural']
                        new_data['Target'] = causal_data['target']
                        new_data['Stance'] = causal_data['causal_counterfactural_label']
                        new_data['llm_stance'] = causal_data['llm_stance']
                        new_data['llm_rationale'] = causal_data['llm_rationale']
                        write_causal_all_datas.append(new_data)
                    else:
                        error_num += 1
                        print(error_num)

                    if f"{data['Tweet']}:{data['Target']}" in non_causal_data2idx:
                        non_causal_data = non_causal_datas[non_causal_data2idx[f"{data['Tweet']}:{data['Target']}"]]
                        new_data = {}
                        new_data['Tweet'] = non_causal_data['non_causal_counterfactural_text']
                        new_data['Target'] = non_causal_data['non_causal_counterfactural_target']
                        new_data['Stance'] = non_causal_data['label']
                        new_data['llm_stance'] = non_causal_data['llm_stance']
                        new_data['llm_rationale'] = non_causal_data['llm_rationale']
                        write_non_causal_all_datas.append(new_data)
                    else:
                        error_num += 1
                        print(error_num)

                write_causal_path = f"{dataset_config.in_target_cad_data_dir['causal'][llm_name]}/{target_name}/train.csv"
                write_non_causal_path = f"{dataset_config.in_target_cad_data_dir['non_causal'][llm_name]}/{target_name}/train.csv"
                if not os.path.isdir(os.path.dirname(write_causal_path)):
                    os.makedirs(os.path.dirname(write_causal_path))
                if not os.path.isdir(os.path.dirname(write_non_causal_path)):
                    os.makedirs(os.path.dirname(write_non_causal_path))
                save_csv(write_causal_all_datas, write_causal_path)
                save_csv(write_non_causal_all_datas, write_non_causal_path)

        # zero-shot
        for target_name in dataset_config.zero_shot_target_names:
            all_datas = load_csv(f'{dataset_config.zero_shot_data_dir}/{target_name}/train.csv')
            write_causal_all_datas = []
            write_non_causal_all_datas = []
            for data in all_datas:
                if f"{data['Tweet']}:{data['Target']}" in causal_data2idx:
                    causal_data = causal_datas[causal_data2idx[f"{data['Tweet']}:{data['Target']}"]]
                    new_data = {}
                    new_data['Tweet'] = causal_data['causal_sentence_counterfactural']
                    new_data['Target'] = causal_data['target']
                    new_data['Stance'] = causal_data['causal_counterfactural_label']
                    new_data['llm_stance'] = causal_data['llm_stance']
                    new_data['llm_rationale'] = causal_data['llm_rationale']
                    write_causal_all_datas.append(new_data)
                else:
                    error_num += 1
                    print(error_num)

                if f"{data['Tweet']}:{data['Target']}" in non_causal_data2idx:
                    non_causal_data = non_causal_datas[non_causal_data2idx[f"{data['Tweet']}:{data['Target']}"]]
                    new_data = {}
                    new_data['Tweet'] = non_causal_data['non_causal_counterfactural_text']
                    new_data['Target'] = non_causal_data['non_causal_counterfactural_target']
                    new_data['Stance'] = non_causal_data['label']
                    new_data['llm_stance'] = non_causal_data['llm_stance']
                    new_data['llm_rationale'] = non_causal_data['llm_rationale']
                    write_non_causal_all_datas.append(new_data)
                else:
                    error_num += 1
                    print(error_num)

            write_causal_path = f"{dataset_config.zero_shot_cad_data_dir['causal'][llm_name]}/{target_name}/train.csv"
            write_non_causal_path = f"{dataset_config.zero_shot_cad_data_dir['non_causal'][llm_name]}/{target_name}/train.csv"
            if not os.path.isdir(os.path.dirname(write_causal_path)):
                os.makedirs(os.path.dirname(write_causal_path))
            if not os.path.isdir(os.path.dirname(write_non_causal_path)):
                os.makedirs(os.path.dirname(write_non_causal_path))
            save_csv(write_causal_all_datas, write_causal_path)
            save_csv(write_non_causal_all_datas, write_non_causal_path)


if __name__ == "__main__":
    # gpt, llama
    # causal, non_causal
    main('gpt', 'causal')
    main('gpt', 'non_causal')
    main('llama', 'causal')
    main('llama', 'non_causal')

    process_counterfactual_augmentation_data('gpt')
    process_counterfactual_augmentation_data('llama')