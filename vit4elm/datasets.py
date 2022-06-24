import json
import pandas as pd
import os
import numpy as np
from transformers import ViTFeatureExtractor
from torch.utils.data import Dataset
from torchvision.io import read_image


class IsingData(Dataset):
    '''
    This class serves as the PyTorch dataset of the processed images from `data_processer.py` script. Because stratified sampling
    is leveraged within the `DataProcessor`, this class loads the train, test, and validation sets immediately.

    params:
        - experiment_json_path (str): path to `experiments.json`. Will be found in `data_dir`.
        - train_test_validation (str): takes in either 'train', 'test', or 'validation' and will parse out
        the respective dataset specified. if left as default, None, it will return the entire set.
        - feature_extractor: take in HuggingFace's ViTFeatureExtractor to perform necessary transformations on
        images. Is also used for ResNet models as well. Currently, the datasets only support using ViTFeatureExtractor,
        so if you change this parameter, it will break this class.
    '''
    def __init__(
        self,
        experiment_json_path:str,
        train_test_validation:str=None,
        feature_extractor = ViTFeatureExtractor()
        ):
        super().__init__()
        
        # Construct set names for parsing out set
        self.__setnames__ = np.array(['train', 'test', 'validation'])

        # Instantiate path and load experiments json
        self.experiment_json_path = experiment_json_path
        with open(self.experiment_json_path, 'r') as j:
            self.experiments = json.load(j)
        
        # Parse out keys from experiments json
        self.data_dir = self.experiments['data_dir']
        self.num_labels = self.experiments['num_labels']
        self.intervals = np.array(self.experiments['intervals'])

        # Construct path to csv and load:
        self.csv_path = os.path.join(self.data_dir, 'csvs', 'data.csv')
        self.csv = pd.read_csv(self.csv_path)
        
        # Instantiate feature extractor
        self.feature_extractor = feature_extractor
        
        # Parse out set:
        self.train_test_validation = train_test_validation
        if self.train_test_validation:
            self._parse_set()
        

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        # Parse image from csv and load image
        img_path = self.csv.iloc[idx, 0]
        img = read_image(img_path)
        
        # Parse label from csv
        label = self.csv.iloc[idx, 1]
        
        # If the feature_extractor exists, use it on the image:
        if self.feature_extractor:
            img = self.feature_extractor(img, return_tensors='pt')['pixel_values']
        
        # Now return everything
        return img, label
    
    def _parse_set(self):
        '''
        Parse set_label column of `self.csv` to get train/test/validation set.
        '''
        try:
            # 0 if train, 1 if test, 2 if validation
            set_num = int(np.where(self.__setnames__ == self.train_test_validation)[0])
            
            # Now only select the part of the dataframe that is the set.
            self.csv = self.csv[self.csv['set_label'] == set_num].reset_index(drop=True) # Reset index for self.__getitem__()
        except:
            KeyError, 'The only keys allowed for train_test_valdiation is "train", "test", "validation", or None.'