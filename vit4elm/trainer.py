import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# TODO: Create Trainer base class with all params except save_confusion_matrix
class ClassificationTrainer:
    '''
    Trainer class to handle all training, test, and evaluation functionality.
    
    - model (nn.Module): pytorch model architecture to be trained. must be a classification
    model.
    - model_name (str): name of model. will be used in naming the checkpoint.
    - train_dataloader (DataLoader): dataloader containing the training set.
    - test_dataloader (DataLoader): dataloader containing the test set.
    - loss_function (torch loss): loss function for training
    - optimizer (Optimizer): torch.optim Optimizer to use for optimization.
    - learning_rate (float): learning rate for training
    - batch_size (int): batch size of data.
    - num_epochs (int): number of epochs to train.
    - device (torch.device): set device for training. default set to torch.default("cpu").
    - display_step_size (int): how many batches to train on.
    - evaluate (bool): if true, evaluate on validation set.
    - save_confusion_matrix (bool): if true, save confusion matrix.
    '''
    def __init__(
        self, 
        model:nn.Module,
        model_name:str,
        train_dataloader:DataLoader,
        test_dataloader: DataLoader,
        loss_function:nn.Module,
        optimizer:Optimizer,
        learning_rate:float,
        batch_size:int,
        num_epochs:int,
        device:torch.device = torch.device('cpu'),
        display_step_size:int = 100,
        evaluate:bool = True,
        save_confusion_matrix:bool = True
        ):
        self.device = device
        self.model = model.to(self.device)
        self.model_name = model_name
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
        self.test_size = len(self.test_dataloader.dataset)
        self.train_size = len(self.train_dataloader.dataset)
        self.num_batches = len(self.test_dataloader)
        
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.display_step_size = display_step_size
        self.evaluate = evaluate
        self.save_confusion_matrix = save_confusion_matrix
    
    def _get_prediction(self, X:torch.Tensor) -> torch.Tensor:
        '''
        This is the procedure to get a prediction for both training and testing. Will wrap it in this function quickly.

        params:
            - X (torch.Tensor): input data of size [batch_size, 1, 3, 224, 224]
        returns:
            - prediction (torch.Tensor): prediction from model of X
        '''
        # Squeeze:
        X = torch.squeeze(X)

        # Get prediction and set to device
        prediction = self.model(X)
        prediction = prediction.to(self.device)
        
        return prediction

    def _train_loop(self):
        '''
        Main train loop called by `self.train()`
        '''
        for batch_num, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            # Get prediction and calculate loss:
            prediction = self._get_prediction(X)
            loss = self.loss_function(prediction, y)

            # Backprop:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Show progress every display_step_size
            if batch_num % self.display_step_size == 0:
                loss, current = loss.item(), batch_num * len(X)
                print(f"Loss: {loss:>7f}  [{current:>5d}/{self.train_size:>5d}]")
    
    @torch.no_grad()
    def _test_loop(self):
        '''
        Main test loop called by `self.train()`.
        '''
        # Instantiate necessary variables
        test_loss, num_correct = 0, 0
        # Now we run evaluation loop:
        for X, y in self.test_dataloader:
            X, y = X.to(self.device), y.to(self.device)
            # Get prediction:
            prediction = self._get_prediction(X)
            # Get loss
            test_loss += self.loss_function(prediction, y).item()
            # If prediction is correct, add 1
            num_correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
        
        # Get loss per batch and accuracy
        test_loss /= self.num_batches
        num_correct /= self.test_size
        print(f"Test Error: \n Accuracy: {(100*num_correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    def train(self):
        # Main loop for training and testing:
        for e in range(self.num_epochs):
            print(f'Epoch {e} \n ---------------------')
            self._train_loop()
            self._test_loop()
        print('Training has been completed.')
        # Save model:
        model_root_dir = os.path.join(self.train_dataloader.dataset.data_dir, 'models')
        # If root directory doesn't exist, create it
        if not os.path.isdir(model_root_dir):
            os.mkdir(model_root_dir)
        # Create full path:
        model_path = os.path.join(model_root_dir, f'{self.model_name}_checkpoint_{e}.pth')
        # Save model:
        torch.save(self.model.state_dict(), model_path)
    
    @torch.no_grad()
    def evaluate(self, validation_dataloader:DataLoader):
        '''
        Run evaluation on validation set and return results.

        params:
            - validation_dataloader (Dataloader):
            - save_confusion_matrix (bool):
        '''
        all_gt = []
        all_predictions = []
        for X, y in validation_dataloader:
            # Set to device:
            X, y = X.to(self.device), y.to(self.device)
            # Get prediction probabilites:
            predictions = self.model(X)
            # Get labels:
            predictions_labels = predictions.argmax(1)
            # Save all values:
            all_gt.append(y.tolist())
            all_predictions(predictions_labels.tolist())
        
        # Flatten:
        all_gt = np.array(all_gt).flatten()
        all_predictions = np.array(all_predictions).flatten()
        
        # Get confusion matrix and visualize:
        cm = confusion_matrix(all_gt, all_predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()