import random
from typing import Tuple
from pathlib import Path

import pandas as pd
from omegaconf.dictconfig import DictConfig

from torch.utils.data import Dataset

from datasets import dataset_zoo
from utils.train_utils import set_seed

class BaseDatasetMaker:
    def __init__(self, 
                 config: DictConfig,) -> None:
        set_seed(config.random_seed)
        self.data_pth = sorted(Path(config.data.data_pth).rglob('*.csv'))

        self.dataset_class = getattr(dataset_zoo, config.dataset.name)
        self.dataset_params = config.dataset.params

        self.split_mode = config.split.name
        self.split_params = config.split.params


    def _get_random_split(self) -> Tuple[list | None]:
        random.shuffle(self.data_pth)
        split_ratio = self.split_params.split_ratio
        n_total = len(self.data_pth)

        if self.split_params.with_test:
            n_train, n_valid = int(n_total * split_ratio[0]), int(n_total * split_ratio[1])
            train_pth, valid_pth, test_pth = self.data_pth[:n_train], self.data_pth[n_train:n_train+n_valid], self.data_pth[n_train+n_valid:]
        else:
            n_train = int(n_total * split_ratio[0])
            train_pth, valid_pth = self.data_pth[:n_train], self.data_pth[n_train:]
            test_pth = None
        
        return train_pth, valid_pth, test_pth


    def _get_split(self) -> Tuple[list | None]:
        if self.split_mode == 'random':
            train_pth, valid_pth, test_pth = self._get_random_split()
        else:
            raise NotImplementedError

        return train_pth, valid_pth, test_pth


    def get_datasets(self) -> Tuple[Dataset | None]:
        train_pth, valid_pth, test_pth = self._get_split()
        trainset = self.dataset_class(train_pth, **self.dataset_params, is_valid=False)
        validset = self.dataset_class(valid_pth, **self.dataset_params, is_valid=True)
        testset = self.dataset_class(test_pth, **self.dataset_params, is_valid=True) if test_pth is not None else None

        return trainset, validset, testset




class ClsDatasetMaker(BaseDatasetMaker):
    def __init__(self, 
                 config: DictConfig,) -> None:
        set_seed(config.random_seed)
        self.data_pth = config.data.data_pth

        self.dataset_class = getattr(dataset_zoo, config.dataset.name)
        self.dataset_params = config.dataset.params

        self.split_mode = config.split.name
        self.split_params = config.split.params

        self.metadata = pd.read_csv(config.data.metadata_pth)
        self.label = self.metadata['label'].tolist()
        self.label_map = {k:idx for idx, k in enumerate(sorted(set(self.label)))}


    def _get_random_split(self) -> Tuple[list | None]:
        raise NotImplementedError
        # indices = list(range(len(self.data_pth)))
        # random.shuffle(indices)
        
        # self.data_pth = [self.data_pth[i] for i in indices]
        # self.label = [self.label[i] for i in indices]
        
        # split_ratio = self.split_params.split_ratio
        # n_total = len(self.data_pth)

        # if self.split_params.with_test:
        #     n_train, n_valid = int(n_total * split_ratio[0]), int(n_total * split_ratio[1])
        #     train_pth, valid_pth, test_pth = self.data_pth[:n_train], self.data_pth[n_train:n_train+n_valid], self.data_pth[n_train+n_valid:]
        #     train_label, valid_label, test_label = self.label[:n_train], self.label[n_train:n_train+n_valid], self.label[n_train+n_valid:]
        # else:
        #     n_train = int(n_total * split_ratio[0])
        #     train_pth, valid_pth = self.data_pth[:n_train], self.data_pth[n_train:]
        #     train_label, valid_label = self.label[:n_train], self.label[n_train:]
        #     test_pth, test_label = None, None
        
        # train_info, valid_info, test_info = (train_pth, train_label), (valid_pth, valid_label), (test_pth, test_label)

        # return train_info, valid_info, test_info


    def _get_stratified_split(self) -> Tuple[list | None]:
        train_df = self.metadata[self.metadata['split']=='train']
        valid_df = self.metadata[self.metadata['split']=='valid']

        if self.split_params.with_test:
            test_df = self.metadata[self.metadata['split']=='test']
        else:
            test_df = None

        return train_df, valid_df, test_df


    def _get_split(self) -> Tuple[list | None]:
        if self.split_mode == 'random':
            raise NotImplementedError
            # train_info, valid_info, test_info = self._get_random_split()

        elif self.split_mode == 'stratified':
            train_info, valid_info, test_info = self._get_stratified_split()

        else:
            raise NotImplementedError

        return train_info, valid_info, test_info


    def get_datasets(self) -> Tuple[Dataset | None]:
        train_info, valid_info, test_info = self._get_split()
        trainset = self.dataset_class(self.data_pth, train_info, self.label_map, **self.dataset_params, is_valid=False)
        validset = self.dataset_class(self.data_pth, valid_info, self.label_map, **self.dataset_params, is_valid=True)
        testset = self.dataset_class(self.data_pth, test_info, self.label_map, **self.dataset_params, is_valid=True) if test_info is not None else None

        return trainset, validset, testset