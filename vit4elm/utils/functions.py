'''Utils for model configuration. Most of these came from https://huggingface.co/blog/fine-tune-vit '''
import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_metric, Dataset, Image

## DATASET RELATED UTILS

def check_for_dir(data_dir:str, dir_name:str):
    '''
    Check if directory within data_dir exists. If it doesn't create it and return the full path.
    '''
    dir_path = os.path.join(data_dir, dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path

# The following functions will be used for stratified sampling:

def load_data_csv(data_dir:str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, 'csvs', 'data.csv'))

def stratified_shuffle(data_dir:str, num_labels:int, test_size:float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Run stratified shuffling method on binned data. Will return the train/test sets.
    '''
    # Load csv:
    df = load_data_csv(data_dir)
    # Instantiate shuffle split:
    sss = StratifiedShuffleSplit(num_labels, test_size=test_size)
    # Get X and y
    X, y = df['image'].to_numpy(), df['label'].to_numpy()
    # Stratified split and construct train/test sets
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    # Return the splits:
    return X_train, y_train, X_test, y_test


def load_dataset_from_csv(path_to_csv:str, apply_transform:bool=True) -> Dataset:
    '''
    Helper function to load validation set which has been saved as CSV file during training.
    
    params:
        - path_to_csv (str): Path to validation csv.
        - apply_transform (bool): Apply default transformation if True. Default set to True.
    
    returns:
        - dataset (Dataset): HuggingFace dataset.
    '''
    df = pd.read_csv(path_to_csv)
    # Load to HF Dataset
    dataset = Dataset.from_pandas(df).cast_column('image', Image())
    if apply_transform:
        dataset = dataset.with_transform(transform)
    return dataset

def transform(example_batch):
    ''' 
    "Perform" feature extraction on data. FE will actually run when the sample is indexed/called. In our case, it will likely be during
    training.
    '''
    feature_extractor = ViTFeatureExtractor(do_resize=True)
    # Converts the PNG file to processed array. This is the notation provided by HF in the link.
    inputs = feature_extractor([x for x in example_batch['image']], return_tensors='pt')
    # If there is an error, check the key for your label.
    inputs['label'] = example_batch['label']
    return inputs

def collate_fn(batch):
    '''Create collated data'''
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

## MODEL/TRAINING RELATED UTILS

def compute_metrics(p):
    '''Compute accuracy by geting MAX probability. Change to np.argmin if minimum prediction is needed.'''
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def load_pretrained_vit_model(num_labels:int, pretrained_path:str='google/vit-base-patch32-224-in21k') -> ViTForImageClassification:
    '''
    Load pretrained ViT for training. Only use this if you want to reproduce results or use a different pretrained model. 
    You can find the list of vision transformers here: https://huggingface.co/models?sort=downloads&search=google%2Fvit-
    
    params:
        - num_labels (int): Number of prediction labels.
        - pretrained_path (str): Either HF URL to ViT for image classification OR path to local pretrained model.
    
    returns:
        - model (ViTForImageClassification): Pretrained ViT model for image classification tasks.
    '''
    model = ViTForImageClassification.from_pretrained(pretrained_path, num_labels = num_labels)
    return model