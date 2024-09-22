import sys
sys.path.append('./')
import logging
from utils.dataset_utils.auto_dataset import AutoDataset
from utils.social_media_texts_utils import split_hash_tag, clean_text
from utils.tools import load_csv

class BaselineDataset(AutoDataset):
    def __init__(self, dataset_config, tokenizer, target_name=None, if_split_hash_tag=True, in_target=None, zero_shot=None, train_data=None, valid_data=None, test_data=None, debug_mode=False):
        super().__init__()
        assert train_data or valid_data or test_data
        assert in_target or zero_shot
        assert not (in_target and zero_shot)
        self.target_name = target_name
        self.if_split_hash_tag = if_split_hash_tag
        self.dataset_config = dataset_config
        
        if in_target:
            if train_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.in_target_data_dir}/{target_name}/test.csv'
            
        elif zero_shot:
            if train_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{target_name}/train.csv'
            elif valid_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{target_name}/valid.csv'
            elif test_data:
                self.data_path = f'{dataset_config.zero_shot_data_dir}/{target_name}/test.csv'

        self.sentences, self.targets, self.labels = self.read_data(self.data_path, debug_mode)
        self.encode_datas(tokenizer, self.sentences, self.targets)
        logging.info(f'Baseline Dataset {dataset_config.dataset_name} loading finished')

    def apply_cleaning(self, sentence):
        if self.if_split_hash_tag:
            sentence = split_hash_tag(sentence.lstrip().rstrip())
        else:
            sentence = sentence.lstrip().rstrip()
        if self.dataset_config.apply_cleaning:
            sentence = clean_text(sentence)
        return sentence

    def read_data(self, path, debug_mode=False):
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
            sentences.append(self.apply_cleaning(data['Tweet']))
            targets.append(data['Target'])
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

    BaselineDataset(
        dataset_config=data_configs['sem16'],
        tokenizer=tokenizer,
        target_name=data_configs['sem16'].in_target_target_names[0],
        if_split_hash_tag=False,
        in_target=True,
        train_data=True,
        debug_mode=True
    )