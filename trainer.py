'''Main training script. Will save validation dataset for evalutaion.''' 
from utils import *
from transformers import TrainingArguments, Trainer
import argparse
import os


if __name__ == '__main__':
    # Set up parser.
    parser = argparse.ArgumentParser(
        description='Train on default model/feature extraction config.'
        )
    parser.add_argument('--data_dir', type=str, help='Path to processed data.') 
    parser.add_argument('--test_split', type=float, help='Ratio of test split (0<=x<=1.)')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs to do (optional.) Every run will have a new shuffled train, test, and validation dataset. Default set to 1 run.')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size (optional.) Deafult set to 16.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Set learning rate of model (optional.) Default set to 2e-4.')
    
    # Instantiate all arguments
    args = parser.parse_args()
    
    for i in range(args.num_runs):
        # Instantiate the preprocessor and set the test size
        # Splitting takes place in this helper function, which means every run will use different data (shuffle=True in train/test split.)
        dataset = load_as_dataset(args.data_dir, args.test_split)

        # Save validation set:
        # Will be validation/validation{i}.csv (validation0.csv, validation1.csv, etc.)
        dataset['validation'].to_csv(os.path.join(args.data_dir, 'validation', f'validation{i}.csv'), index=None) # Drop index column

        # Parse out number of classes:
        labels = dataset['train'].features['label'].names
        num_labels = len(labels)
        
        # Run feature extractor on datset
        processed_data = dataset.with_transform(transform)

        # Load model
        model = load_pretrained_vit_model(num_labels)

        # Set training arguments
        training_args = TrainingArguments(
        output_dir=os.path.join(args.data_dir, 'training_results', f'training_results{i}'),
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=4,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=False
    )
        # Instantiate trainer
        trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=processed_data['train'],
        eval_dataset=processed_data['test']
    )
        # Train model
        train_results = trainer.train()

        # Save model and evaluation metrics
        trainer.save_model(f'models/{args.data_dir}')
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()