'''Utils for model configuration. Most of these came from https://huggingface.co/blog/fine-tune-vit '''
import torch
import numpy as np
from transformers import ViTForImageClassification, ViTFeatureExtractor
from datasets import load_metric, load_dataset, Dataset, DatasetDict

def load_as_dataset(data_dir:str, test_size:float) -> Dataset:
    '''Quick wrapper to load processed data into HF usable dataset'''
    dataset = load_dataset('imagefolder', data_dir = data_dir, split='train')
    dataset = dataset.train_test_split(test_size)
    # TODO: WILL ADD WHEN EVALUATION SCRIPT IS COMPLETE
    # # Now we split the test set in half for validation:
    # test_validation = dataset['test'].train_test_split(0.5)
    # dataset = DatasetDict(
    #     {
    #         'train': dataset['train'],
    #         'test': test_validation['test'],
    #         'valid': test_validation['train']
    #     }
    # )
    return dataset


def transform(example_batch):
    ''' "Perform" feature extraction on data. FE will actually run when the sample is indexed/called.'''
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

def compute_metrics(p):
    '''Compute accuracy by geting MAX probability. Change to np.argmin for miniumum.'''
    metric = load_metric("accuracy")
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


def load_pretrained_vit_model(num_labels:int, pretrained_path:str='google/vit-base-patch32-224-in21k') -> ViTForImageClassification:
    '''
    Load pretrained ViT for training. Only use this if you want to reproduce results or use a different pretrained model. 
    You can find the list of vision transformers here: https://huggingface.co/models?sort=downloads&search=google%2Fvit-
    '''
    model = ViTForImageClassification.from_pretrained(pretrained_path, num_labels = num_labels)
    return model