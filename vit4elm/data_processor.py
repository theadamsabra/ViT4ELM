'''
Script of all helper + core data processor class DataProcessor.

Used exclusively data preprocessing (i.e. data.json -> binned jpgs in data/)
'''

import json
import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


class jsonParser:
    '''
    Parser class to handle relevant data from `data.json` easily.
    
    params:
        - json_path (str): Path to JSON of saved simulation.
    '''
    def __init__(self, json_path:str) -> None:
        self.json_path = json_path

        with open(self.json_path, 'r') as j:
            j = json.load(j)
        
        # Parse settings and data as first level access
        self.settings = j['settings']
        self.data = j['data']

        # Can add more if needed later on.
        self.temp_min = self.settings['tempMin']
        self.temp_max = self.settings['tempMax']
        self.temp_step = self.settings['tempStep']

class nBin:
    '''
    Handles all binning actions including interval creation, data binning, and data alignment.
    
    params:
        - min_ (int/float): Minimum temperature value. Supports both integer and float.
        - max_ (int/float): Maximum temperature value. Supports both integer and float.
        - n_bins (int): Number of bins to split between minimum and maximum.
        - bin_len (float): Length of each bin.
    '''
    def __init__(self, min_, max_, n_bins:int, bin_len:float=None) -> None:
        self.min_ = min_
        self.max_ = max_
        self.n_bins = n_bins
        self.bin_len = bin_len
    
    def _calculate_intervals(self):
        '''Calculate bounds of intervals as an array of shape `(n_bins, 2)`, where the columns represent the lower and upper bound respectively.'''
        self.intervals = np.array([
            [self.min_ + (i * self.bin_len) for i in range(self.n_bins)], # Lower bound
            [self.min_ + ((i + 1) * (self.bin_len)) for i in range(self.n_bins)] # Upper bound
        ]).T

class DataProcessor:
    '''
    Processing class to process saved simulations from https://www.statmechsims.com/models/metropolis to
    friendly format.

    params:
        - json_path (str): path to JSON of saved simulation.
        - data_dir (str): best to be the name of your experiment. experiment directory where all images will be saved.
        - n_bins (int): number of classes for data to be binned in. default is None.
        - custom_intervals (np.ndarray): set custom intervals for inequal boundaries, if desired. MUST be of shape (n_bins, 2) where each row is the
        upper and lower band of each interval. default is None.
        - stratified_shuffle (bool): flag for developing the train/test/validation sets via stratified shuffling. default
        is true.
        - test_size (float): test size for train set. will inevitably be split in half for validation set. Default set 
        to 0.4.
        - img_shape (tuple): shape to resize image for standardization. default set to (224,224)
    '''
    def __init__(
        self, 
        json_path:str, 
        data_dir:str, 
        n_bins:int=None, 
        custom_intervals:np.ndarray=None,
        stratified_shuffle:bool=True,
        test_size:float=0.4
        ) -> None:
        # Do standard initalization:
        self.json_path = json_path
        self.stratified_shuffle = stratified_shuffle
        self.test_size = test_size
        # Instantiate parser
        self.parser = jsonParser(self.json_path) 
        
        # Add n_bins and data_dir
        self.n_bins = n_bins
        self.data_dir = data_dir
        # Ensure it exists:
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

        # Instantiate binner for continuity:
        self.binner = nBin(
            min_ = self.parser.temp_min,
            max_ = self.parser.temp_max,
            n_bins = self.n_bins
        )
        # This allows me to add custom_intervals to binner.intervals
        # It might change later
        self.binner.intervals = custom_intervals
        
        # If both are instantiated:
        if self.binner.intervals is not None and self.n_bins != None:
            # Ensure custom intervals line up with bins
            assert self.binner.intervals.shape[0] == n_bins, 'Number of specified bins must be equal to the number of rows in your custom intervals!'
        
        # If intervals are instantiated and n_bins are not instantiated:
        elif self.binner.intervals is not None and self.n_bins == None:
            # Just set number of bins to the number of rows in your intervals.
            self.n_bins = self.binner.intervals.shape[0]
        
        # If intervals are not instantiated and number of bins are instantiated:
        elif self.binner.intervals is None and self.n_bins != None:
            # Calculate equidistant range for each bin.
            self.bin_len = (self.parser.temp_max - self.parser.temp_min) / self.n_bins
            
            # Instantiate binner properly:
            self.binner = nBin(
                min_ = self.parser.temp_min,
                max_ = self.parser.temp_max,
                n_bins = self.n_bins,
                bin_len = self.bin_len
            )

            # Create interval bounds. Will be saved as self.binner.intervals of shape (n_bins, 2)
            self.binner._calculate_intervals()
        
        # Otherwise, neither are instantiated. You're gonna need at least one.
        else:
            assert (custom_intervals is not None and n_bins != None), 'Instantiate at least one parameter! Either n_bins or custom_intervals.'
    
    def _make_bins(self, data_dir_subdirectory:str):
        '''
        Make bin0...bin{n-1} directories.
        
        params:
            - data_dir_subdirectory (str): Path to subdirectory to create bins within.
        '''
        # Create new bins:
        for bin in range(self.n_bins):
            os.mkdir(os.path.join(data_dir_subdirectory, f'bin{bin}'))

    def _check_dir_exists(self, dir_name:str):
        '''
        Check if data directory exists. If it does, remove the directory tree and create a new one for bins to be inside. If it does not exist,
        create it and bins.

        params:
            - dir_name (str): Name of directory to see if it exists.
        '''
        # Check if data directory within experiment exists and remove it entirely.
        data_dir_subdirectory = os.path.join(self.data_dir, dir_name)
        if os.path.isdir(data_dir_subdirectory): 
            shutil.rmtree(data_dir_subdirectory)
        os.mkdir(data_dir_subdirectory)
    
    def _colormap(self, array_:np.ndarray):
        '''
        Convert array to image data
        
        params:
            - array_ (np.ndarray): 100x100 image array.
        '''
        # TODO: Create new color mapping for other data
        cmap = {
            1: (255,255,255),
            -1: (0,0,0)
        }
        data = [cmap[spin] for spin in array_.flatten()]
        img = Image.new('RGB', (100,100), 'white')
        img.putdata(data)
        return img
    
    def _save_dataframe(self, image_col:list, labels:list, save_path:str):
        '''
        Quick function to save dataframes.
        
        params:
            - image_col (list): Data for image column.
            - labels (list): Data for labels.
            - save_path (str): Path to save the dataframe.
        '''
        pd.DataFrame({
            'image': image_col,
            'label': labels
        }).to_csv(save_path, index=None)
    
    def shuffle_and_save(self):
        '''
        Leverage stratified shuffling to the main dataframe and save train/test set. The new csv that will be saved will contain
        3 columns: path to images `image`, their corresponding label `label`, and which set they belong to `set_label`.

        `set_label` = 0 is train set
        `set_label` = 1 is test set
        `set_label` = 2 is validation set
        '''
        # Load dataframe as well as stratified shuffler
        df = pd.read_csv(self.csv_path)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size)
        # Use numpy versions to get split labels:
        X, y = df['image'].to_numpy(), df['label'].to_numpy()
        # Set all (train/test/validation) set labels to be 0. This will be all train.
        df['set_label'] = len(df) * [0]
        # Stratified split and construct train/test sets
        # We have to loop as sss.split() returns a generator object.
        for _, test_index in sss.split(X, y):
            # Now find and replace the test set with set_label = 1
            df.loc[test_index, 'set_label'] = 1
        # Parse out the test set and split in half.
        test_df = df[df['set_label'] == 1]
        val = test_df.sample(frac=0.5)
        # Use those indices of validation to change the labels
        df.loc[val.index, 'set_label'] = 2
        # Now we save the new dataframe:
        df.to_csv(self.csv_path, index=None)

    def process(self):
        '''
        Process function to be used by end-user.
        '''
        # Check if data directory exists to empty out. If it does not exist, create it.
        self._check_dir_exists(dir_name='data')
        # Add bin0 ... bin(n-1)
        self._make_bins(os.path.join(self.data_dir, 'data'))
        # Iterate through data and match
        # Will add a new key called bin_number when specified bin is created.
        # After this loop, a csv will be created:
        save_paths = []
        labels_ = []
        for data_info in self.parser.data:
            # Parse out temp
            temp = data_info['temp']
            # Save data array as image to data/bin_number
            array_ = np.array([float(x)for x in data_info['spinArray'].split(',')]).reshape(100,100)
            # Some handling is done to convert to color mapping
            img = self._colormap(array_)
            # Loop to through bins to find bin number:
            for bin_number in range(self.n_bins):
                if self.binner.intervals[bin_number, 0] <= temp <= self.binner.intervals[bin_number, 1]:
                    data_info['bin_number'] = bin_number
            # Construct save path
            save_path = os.path.join(self.data_dir, 'data', f'bin{data_info["bin_number"]}', f'{data_info["timestamp"]}.jpg')
            # Save array as JPEG
            img.save(save_path)
            # Add save paths and bin number to list for dataframe
            save_paths.append(save_path)
            labels_.append(data_info["bin_number"])
        # Construct/save dataframe
        self._check_dir_exists('csvs')
        self.csv_path = os.path.join(self.data_dir, 'csvs', 'data.csv')
        self._save_dataframe(save_paths, labels_, self.csv_path)
        # Shuffle data csv and save
        if self.stratified_shuffle:
            self.shuffle_and_save()
        # Write JSON for useful information to pass into trainer.
        experiments = {
        'data_dir': self.data_dir,
        'num_labels': self.n_bins,
        'intervals': self.binner.intervals.tolist(),
        'test_size': self.test_size
        }
        experiment_json_path = os.path.join(self.data_dir, 'experiments.json')
        with open(experiment_json_path, 'w') as j:
            json.dump(experiments, j)