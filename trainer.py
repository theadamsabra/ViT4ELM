'''Main training script. Will save validation dataset for evalutaion.''' 
from cgi import test
from tabnanny import check
from utils import *
from transformers import TrainingArguments, Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import os
import json
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Set up parser.
    parser = argparse.ArgumentParser(
        description='Train on default model/feature extraction config.'
        )
    parser.add_argument('--data_dir', type=str, help='Path to processed data.')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of training runs to do (optional.) Every run will have a new shuffled train, test, and validation dataset. Default set to 1 run.')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size (optional.) Deafult set to 16.')
    parser.add_argument('--lr', type=float, default=2e-4, help='Set learning rate of model (optional.) Default set to 2e-4.')
    parser.add_argument('--eval', type=bool, default=True, help='Run evaluation if True, just train if False.')
    
    # Instantiate all arguments
    args = parser.parse_args()

    # Load experiments json for additional experiment information:
    with open(os.path.join(args.data_dir, 'experiments.json')) as e:
        experiment = json.load(e)
    
    # Parse out number of classes:
    num_labels = experiment['num_labels']

    # Load train, test, and validation sets
    train = load_dataset_from_csv(os.path.join(args.data_dir, 'csvs', 'train.csv'))
    test = load_dataset_from_csv(os.path.join(args.data_dir, 'csvs', 'test.csv'))
    validation = load_dataset_from_csv(os.path.join(args.data_dir, 'csvs', 'validation.csv'))

    for i in range(args.num_runs):
        # Load model
        model = load_pretrained_vit_model(num_labels)

        # Check for training results path:
        training_results_path = check_for_dir(args.data_dir, 'training_results')
        
        # Set training arguments
        training_args = TrainingArguments(
        output_dir=os.path.join(training_results_path, f'training_results{i}'),
        per_device_train_batch_size=args.batch_size,
        evaluation_strategy="steps",
        num_train_epochs=4,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=args.lr,
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True
        )

        # Instantiate trainer
        trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train,
        eval_dataset=test
    )
        # Train model
        train_results = trainer.train()

        # Save model and evaluation metrics
        save_model_path = check_for_dir(args.data_dir, 'models')
        trainer.save_model(os.path.join(save_model_path, f'model_run_{i}'))
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        # Run evaluation
        if args.eval:
            # Create new directory for evaluation:
            evaluation_output = check_for_dir(args.data_dir, 'evaluations')
            evaluation_output = os.path.join(evaluation_output, f'evaluation_run_{i}')
            
            # Run prediction:
            outputs = trainer.predict(validation)
            
            # Get confusion matrix
            y_true = outputs.label_ids
            y_pred = outputs.predictions.argmax(1)
            cm = confusion_matrix(y_true, y_pred) # Outputs numpy array of confusion matrix
            
            # Save numpy array of confusion matrix:
            cm_path = os.path.join(evaluation_output, f'confusion_array')
            np.save(cm_path, cm)

            # Visualize confusion matrix:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            plt.show()