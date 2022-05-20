import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

if __name__ == '__main__':
    # Set up CLI parser
    parser = argparse.ArgumentParser(
        description='Run evaluation on all validation sets.'
        )
    parser.add_argument('--data_dir', type=str, help='Path to processed data.') 
    parser.add_argument('--validation_num', type=int, help='Validation set number to process checkpoint on.')
    parser.add_argument('--checkpoint_num', type=int, help='Checkpoint number to process')
    parser.add_argument('--num_labels', type=int, help='Number of labels')
    parser.add_argument('--visualize_cm', type=bool, default=True, help='Visualize confusion matrix.')
    args = parser.parse_args()
    
    # Load model
    model = load_pretrained_vit_model(args.num_labels, os.path.join(args.data_dir, 'training_results', f'training_results{args.validation_num}', f'checkpoint-{args.checkpoint_num}'))
    
    # Load validation set as HF dataset
    validation_dir = check_for_dir(args.data_dir, 'validation')
    validation = load_dataset_from_csv(os.path.join(validation_dir, f'validation{args.validation_num}.csv'))
    # Perform feature extraction on validation
    validation = validation.with_transform(transform)

    # Load dummy trainer but to run evaluation
    evaluation_output = os.path.join(args.data_dir, 'evaluation')
    trainer = quick_load_trainer(model, evaluation_output)

    # Run prediction:
    outputs = trainer.predict(validation)

    # Get confusion matrix
    y_true = outputs.label_ids
    y_pred = outputs.predictions.argmax(1)
    cm = confusion_matrix(y_true, y_pred) # Outputs numpy array of confusion matrix
    
    # Save numpy array:
    cm_path = os.path.join(evaluation_output, f'confusion_checkpoint_{args.checkpoint_num}_validation{args.validation_num}')
    np.save(cm_path, cm)
    
    # Visualize if set to true.
    if args.visualize_cm:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()