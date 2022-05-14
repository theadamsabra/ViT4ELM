'''Utils for model configuration. Most of these came from https://huggingface.co/blog/fine-tune-vit '''
import torch
import numpy as np
import pandas as pd
from transformers import ViTForImageClassification, ViTFeatureExtractor, Trainer, TrainingArguments
from datasets import load_metric, load_dataset, Dataset, DatasetDict, Image

## DATASET RELATED UTILS

def load_as_dataset(data_dir:str, test_size:float) -> DatasetDict:
    '''
    Quick wrapper to load processed data into HF usable dataset
    
    params:
        - data_dir (str): Directory to data
        - test_size (float): Size of test set. Will then be split in half for validation.
    
    returns:
        - dataset (DatasetDict): HF Wrapper of Dataset.
    '''
    dataset = load_dataset('imagefolder', data_dir = data_dir, split='train')
    dataset = dataset.train_test_split(test_size)
    # Now we split the test set in half for validation:
    test_validation = dataset['test'].train_test_split(0.5)
    # Thanks hf forums
    dataset = DatasetDict(
        {
            'train': dataset['train'],
            'test': test_validation['test'],
            'validation': test_validation['train']
        }
    )
    return dataset

def load_dataset_from_csv(path_to_csv:str) -> Dataset:
    '''
    Helper function to load validation set which has been saved as CSV file during training.
    
    params:
        - path_to_csv (str): Path to validation csv.
    
    returns:
        - dataset (Dataset): HuggingFace dataset.
    '''
    df = pd.read_csv(path_to_csv)
    # Process image column such that "image" column is the path to the jpg.
    processed_col = [img_col.split('\'path\':')[1].strip('}').strip().strip('\'') for img_col in df['image']]
    # Replace the column
    df['image'] = processed_col
    # Load to HF Dataset
    dataset = Dataset.from_pandas(df).cast_column('image', Image())
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

## EVAL RELATED UTILS 

def quick_load_trainer(model:ViTForImageClassification, output_dir:str) -> Trainer:
    '''
    Load a pseudo Trainer class. Although it looks identical to the Trainer class in the trainer.py script, there are some differences as to HOW
    this Trainer is used. Notice, the Trainer does not have data fed into it. This is because it will be used for EVALUATION.

    params:
        - model (ViTForImageClassification): Pretrained model either from HuggingFace or from prior training.
        - output_dir (str): Output directory for saving evaluation.
    
    returns:
        - trainer (Trainer): HF Trainer. Used for both training and evaluation.
    '''
    # Set training arguments
    # These arguments don't really matter. However, loading them all in makes processing everything smooth.
    # This "Trainer" will be used for prediction on the validation set.
    training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=4,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to='tensorboard',
    load_best_model_at_end=False
)
    # Instantiate trainerViTForImageClassification
    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)
    return trainer