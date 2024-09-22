import sys
sys.path.append('./')
import logging
from utils.dataset_utils.auto_dataset import AutoDataset
from utils.social_media_texts_utils import split_hash_tag, clean_text
from utils.tools import load_csv

class RationaleDataset(AutoDataset):
    def __init__(self, args, dataset_config, llm, tokenizer, target_name=None, in_target=None, zero_shot=None, train_data=None, valid_data=None, test_data=None, debug_mode=False):
        super().__init__()
        assert train_data or valid_data or test_data
        assert in_target or zero_shot
        assert not (in_target and zero_shot)
        self.target_name = target_name
        self.if_split_hash_tag = args.if_split_hash_tag
        self.dataset_config = dataset_config
        
        if in_target:
            if train_data:
                self.data_path = f'{dataset_config.in_target_llm_rationale_data_dir[llm]}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.in_target_llm_rationale_data_dir[llm]}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.in_target_llm_rationale_data_dir[llm]}/{target_name}/test.csv'
        elif zero_shot:
            if train_data:
                self.data_path = f'{dataset_config.zero_shot_llm_rationale_data_dir[llm]}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.zero_shot_llm_rationale_data_dir[llm]}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.zero_shot_llm_rationale_data_dir[llm]}/{target_name}/test.csv'

        self.sentences, self.targets, self.labels = self.read_data(self.data_path, debug_mode)
        self.encode_datas(tokenizer, self.sentences, self.targets)
        logging.info(f'Rationale Dataset {dataset_config.dataset_name} loading finished')

    def apply_cleaning(self, sentence):
        if self.if_split_hash_tag:
            sentence = split_hash_tag(sentence.lstrip().rstrip())
        else:
            sentence = sentence.lstrip().rstrip()
        if self.dataset_config.apply_cleaning:
            sentence = clean_text(sentence)
        return sentence

    def read_data(self, path, debug_mode=False):
        self.raw_sentences = []
        self.raw_targets = []
        sentences = []
        targets = []
        labels = []
        label_num = {}
        for label_name in self.dataset_config.label2idx.keys():
            label_num[label_name] = 0
        all_datas = load_csv(path) 

        if debug_mode:
            all_datas = all_datas[:200]
        for data in all_datas:
            sentences.append(f"Text: {self.apply_cleaning(data['Tweet'])} Analysis: {data['llm_rationale']}")
            self.raw_sentences.append(data['Tweet'])
            targets.append(data['Target'])
            self.raw_targets.append(data['Target'])
            labels.append(self.dataset_config.label2idx[data['Stance']])
            label_num[data['Stance']] += 1
        logging.info(f'loading data {len(sentences)} from {path}')
        logging.info(f'label num ' + ' '.join([f'{k}: {v}' for k,v in label_num.items()]))
        return sentences, targets, labels


if __name__ == '__main__':
    from transformers import AutoTokenizer
    from utils.dataset_utils.data_config import data_configs
    transformer_tokenizer_name = 'model_state/roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(transformer_tokenizer_name)
    tokenizer.model_max_length=tokenizer.max_model_input_sizes['/'.join(transformer_tokenizer_name.split('/')[1:])]

    class Args():
        if_split_hash_tag = True

    args = Args()

    RationaleDataset(
        dataset_config=data_configs['vast'],
        llm='gpt',
        tokenizer=tokenizer,
        target_name=data_configs['vast'].zero_shot_target_names[0],
        if_split_hash_tag=False,
        zero_shot=True,
        train_data=True,
        debug_mode=False
    )