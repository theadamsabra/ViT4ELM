# Applications of Vision Transformers for Exotic Lattice Models

## About

Applications of Vision Transformers to Exotic Lattice Models. This repository contains all of the finetuned models as well as core scripts to reproduce results and experiment on new data. The experimentation suite that this repository builds on top of can be found [in this website](https://www.statmechsims.com/). The goal of this repository is to provide an end-to-end experimentation and evaulation of Vision Transfomers on these types of systems.

Raising issues are encouraged so we know what features to prioritize. We want to inevitably work towards predicting regions of interest using masked patch prediction, so this is our top most priority on our road map.

## Documentation

### Format Data from JSON to User Friendly images

To process the data, ensure you have a `data.json` from [statmechsims](https://www.statmechsims.com/). This json will contain all of your simulation data.

The result of the above will lead to a new directory named `af_ising` with the following structure:

```console
python3 data_handle.py --json_path="/path/to/data.json" --n_bins=4 --data_dir"af_ising"
```

Binning is set by the temperature value, all of equal length between the minimum and maximum temperature.

### NOTE: USE THE CLI IF YOU WANT TO USE EQUIDISTANT TEMPERATURE BOUNDARIES. ALSO, PLEASE MAKE SURE THE DIRECTORY (`af_ising` IN THIS EXAMPLE) DOES NOT EXIST. IF THE DIRECTORY EXISTS, IT WILL BE REMOVED AND REMADE

**If you would like to add custom intervals, simply run the following in a Jupyter notebook or Python script:**

```python
# Import DataProcessor class
from data_handle import DataProcessor
import numpy as np

# Instantiate necessary parameters.
json_path = '/path/to/data.json'
data_dir = 'af_ising'

# Please ensure that the shape of custom_intervals is (n_bins, 2)
custom_intervals = np.array([
    [0, 2],
    [2, 3.33],    # Notice the funky boundary lengths
    [3.33, 3.55], 
    [3.55, 4]
])

# Instantiate DataProcessor
processor = DataProcessor(json_path, data_dir, custom_intervals=custom_intervals)

# Let the processor process!
processor.process()
```

Should you decide to use custom intervals, please ensure the min/max temperatures (0 and 4, in this example) line up with your simulation experiments.

The result of either the following CLI call or running in a notebook/script will lead to a directory with the following structure.

```console
af_ising/
├── bin0/ 
├── bin1/
├── bin2/
└── bin3/ 
```

The only difference between these is the boundaries that determine which image goes into which bin.

## Training the Model

To train on the pretrained ViT Base 32 model with default configurations for feature extraction, run the following command.

```console
python3 trainer.py --data_dir="af_ising" --test_split=0.4 --num_runs=1 --batch_size=16 --lr=2e-4
```

The results of the following run will be in your `data_dir`:

```console
af_ising/
├── bin0/ 
├── bin1/
├── bin2/
├── bin3/
├── training_results/
│   └── training_results0
└── validation/
    └── validation0.csv
```

The necessary parameters you ***must*** input are `data_dir` and `test_split`. The last 3 parameters (`num_runs`, `batch_size`, and `lr`) are optional. The default values for those are `1`, `16`, and `2e-4`, respectively.

Make sure that the `data_dir` parameter from both `data_handle` and `trainer` CLI calls are the same and there should be no issue. Furthermore, I highly recommend you increase the `num_runs` to at least 5 to have a solid evaluation confidence interval.

## To Do

Low Hanging Fruit:

- [ ] Add nice cover image to repo
- [x] Add train/test/validation + saving validation set mechanism
- [x] Load model
- [x] Run on validation set
- [x] Get confusion matrix
- [ ] Wrap functions into eval script

Bigger Stuff:

- [ ] Head wise visualizations
- [ ] Add masker to images
