# Applications of Vision Transformers for Exotic Lattice Models

## About

Applications of Vision Transformers to Exotic Lattice Models. This repository contains all of the finetuned models as well as core scripts to reproduce results and experiment on new data. The experimentation suite that this repository builds on top of can be found [in this website](https://www.statmechsims.com/). The goal of this repository is to provide an end-to-end experimentation and evaulation of Vision Transfomers on these types of systems.

Raising issues are encouraged so we know what features to prioritize. We want to inevitably work towards predicting regions of interest using masked patch prediction, so this is our top most priority on our road map.

## Documentation

### Install Requirements

Nothing new here. Just make sure to create a new Python environment and run the following:

```console
pip install -r requirements.txt
```

### Format Data from JSON to User Friendly images

To process the data, ensure you have a `data.json` from [statmechsims](https://www.statmechsims.com/). This json will contain all of your simulation data. Say you have the following directory:

```console
af_ising/
└── data.json
```

Running the below command will add to your main experiment directory `af_ising` with the following structure:

```console
python3 data_handle.py --json_path="af_ising/data.json" --n_bins=4 --data_dir"af_ising" --stratified_shuffle=True --test_size=0.4
```

Binning is set by the temperature value, all of equal length between the minimum and maximum temperature.

### NOTE: USE THE CLI IF YOU WANT TO USE EQUIDISTANT TEMPERATURE BOUNDARIES. ALSO, PLEASE MAKE SURE THE DIRECTORY (`af_ising` IN THIS EXAMPLE) DOES NOT EXIST. IF THE DIRECTORY EXISTS, IT WILL BE REMOVED AND REMADE

**If you would like to add custom intervals, simply run the following in a Jupyter notebook or Python script:**

```python
# Import DataProcessor class
from data_handle import DataProcessor
import numpy as np

# Instantiate necessary parameters.
json_path = 'af_ising/data.json'
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
├── csvs
│   ├── data.csv
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── data
│   ├── bin0
│   ├── bin1
│   ├── bin2
│   └── bin3
├── data.json
└── experiments.json
```

The only difference between these is the boundaries that determine which image goes into which bin based off of the temperature. Again, this is assuming you want to bin into 4 bins.

## Training the Model

To train on the pretrained ViT Base 32 model with default configurations for feature extraction, run the following command.

```console
python3 trainer.py --data_dir="af_ising" --num_runs=1 --batch_size=16 --lr=2e-4 --eval=True
```

The necessary parameters you ***must*** input is `data_dir`. 

The last 4 parameters (`num_runs`, `batch_size`, `lr`, `eval`) are optional. The default values for those are `1`, `16`, `2e-4`, and `True` respectively. Make sure that the `data_dir` parameter from both `data_handle` and `trainer` CLI calls are the same and there should be no issue. Furthermore, I highly recommend you increase the `num_runs` to at least 5 to construct solid evaluation. The instantiated ViT weights are randomized after all.

Evaluation will take place on a per run basis. There currently does not exist a seperate script that does evaluation independently, so keep `eval` to be `True` until it is allowed seperately. During evaluation, a confusion matrix will be saved as a numpy array. Furthermore, `pyplot` will display the confusion matrix. So, if you want to save the image, you can.

The results of the following run will be in your `data_dir`:

```console
datasets/af_ising
├── csvs
│   ├── data.csv
│   ├── test.csv
│   ├── train.csv
│   └── validation.csv
├── data
│   ├── bin0
│   ├── bin1
│   ├── bin2
│   └── bin3
├── data.json
├── experiments.json
├── models
│   └── model_run_0
└── training_results
    └── training_results0
```

## To Do

Smaller Stuff:

- [ ] Better model loading
- [ ] Add cover image for repo

Bigger Stuff:

- [ ] Head wise visualizations
- [ ] Add masker to images