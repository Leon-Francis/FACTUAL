from utils.dataset_utils.dataset_configs.p_stance_config import PStanceConfig
from utils.dataset_utils.dataset_configs.vast_config import VASTConfig
from utils.dataset_utils.dataset_configs.sem16_config import Sem16Config

data_configs = {
    'sem16': Sem16Config,
    'p_stance': PStanceConfig,
    'vast': VASTConfig
}

from utils.dataset_utils.baseline_dataset import BaselineDataset

datasets = {
    'baseline': BaselineDataset
}

from utils.dataset_utils.rationale_dataset import RationaleDataset
from utils.dataset_utils.counterfactual_augmented_dataset import CounterfactualAugmentationDataset

factual_datasets = {
    'rationale': RationaleDataset,
    'cad': CounterfactualAugmentationDataset
}