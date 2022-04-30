'''
Script of all helper + core data processor class DataProcessor.

Used exclusively data preprocessing (i.e. data.json -> binned jpgs in data/)
'''

import json
import numpy as np
import os
import shutil
import argparse
from PIL import Image

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
    '''Handles all binning actions including interval creation, data binning, and data alignment.'''
    def __init__(self, min_, max_, n_bins:int, bin_len:float) -> None:
        '''
        params:
            - min_ (int, float): minimum temperature
            - max_ (int, float): maximum temperature
            - n_bins (int): n_bins
            - bin_len (float): float
            '''
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
    Handling class to process saved simulations from https://www.statmechsims.com/models/metropolis to
    friendly format.
    '''
    def __init__(self, json_path:str, n_bins:int, data_dir='data') -> None:
        '''
        params:
            - json_path (str): Path to JSON of saved simulation.
            - n_bins (int): Number of classes for data to be binned in.
            - data_dir (str): Best to be the name of your experiment. Experiment directory where all images will be saved.
        '''
        self.json_path = json_path
        # Instantiate parser
        self.parser = jsonParser(self.json_path) 
        self.n_bins = n_bins
        self.data_dir = data_dir
        # Calculate equidistant range for each bin. 
        self.bin_len = (self.parser.temp_max - self.parser.temp_min) / self.n_bins
        # Instantiate bin creation.
        self.binner = nBin(
            min_ = self.parser.temp_min,
            max_ = self.parser.temp_max,
            n_bins = self.n_bins,
            bin_len = self.bin_len
        )
        # Create interval bounds. Will be saved as self.binner.intervals of shape (n_bins, 2)
        self.binner._calculate_intervals()
    
    def _check_data_dir_exists(self):
        '''
        Check if data directory exists. If it does, remove the directory tree and create a new one for bins to be inside. If it does not exist,
        create it and bins.
        '''
        if os.path.isdir(self.data_dir): # Check if it exists and remove it entirely.
            shutil.rmtree(self.data_dir)
        os.mkdir(self.data_dir)
        for bin in range(self.n_bins):
            os.mkdir(f'{self.data_dir}/bin{bin}')
    
    def _colormap(self, array_:np.ndarray):
        '''
        Convert array to image data
        '''
        cmap = {
            1: (255,255,255),
            -1: (0,0,0)
        }
        data = [cmap[spin] for spin in array_.flatten()]
        img = Image.new('RGB', (100,100), 'white')
        img.putdata(data)
        return img

    def process(self):
        '''Process function to be used by end-user.'''
        # Check if data directory exists to empty out. If it does not exist, create it.
        self._check_data_dir_exists()
        # Iterate through data and match:
        # Will add a new key called bin_number when specified bin is created.
        # After this loop, a csv will be created:
        for data_info in self.parser.data:
            temp = data_info['temp']
            for bin_number in range(self.n_bins):
                if self.binner.intervals[bin_number, 0] <= temp <= self.binner.intervals[bin_number, 1]:
                    data_info['bin_number'] = bin_number
            # Save data array as image to data/bin_number:
            array_ = np.array([float(x)for x in data_info['spinArray'].split(',')]).reshape(100,100)
            # Some handling is done to convert to color mapping:
            img = self._colormap(array_)
            save_path = f'{self.data_dir}/bin{data_info["bin_number"]}/{data_info["timestamp"]}.jpg'
            # Save numpy array
            img.save(save_path)

if __name__ == '__main__':
    # Set up parser.
    parser = argparse.ArgumentParser(
        description='Process data JSON to HF friendly format.'
        )
    parser.add_argument('--json_path', type=str, help='Path to JSON')
    parser.add_argument('--n_bins', type=int, help='Number of bins to bin across start/end temperature.')
    parser.add_argument('--data_dir', type=str, help='Path to save binned data. Best it is your simulation name.')

    args = parser.parse_args()

    # Instantiate processor and run it.
    processor = DataProcessor(args.json_path, args.n_bins, args.data_dir)
    processor.process()