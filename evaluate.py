from utils import *
import torch
import os


if __name__ == '__main__':
    DATASET = 'af_ising'
    NUM_RUN = 'training_results0'
    VALIDATION_CSV = 'datasets/af_ising/validation0.csv'

    output_dir = os.path.join('datasets', DATASET)
    batch_size = 16
    lr = 2e-4

    training_args = load_training_args(output_dir, batch_size, lr)
    model = load_pretrained_vit_model(4, os.path.join('datasets', DATASET, NUM_RUN, 'checkpoint-100')) # Will need function to search for checkpoint num
    validation = load_dataset_from_csv(VALIDATION_CSV, apply_transform = True)

    ds = load_as_dataset(f'datasets/{DATASET}', 0.4)
    trainer = load_standard_trainer(model, training_args, ds)

    metrics = trainer.evaluate(validation)
    print(metrics)