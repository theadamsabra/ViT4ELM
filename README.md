# Applications of Vision Transformers for Exotic Lattice Models

## About

Applications of Vision Transformers to Exotic Lattice Models. This repository contains all of the finetuned models as well as core scripts to reproduce results and experiment on new data. The experimentation suite that this repository builds on top of can be found [in this website](https://www.statmechsims.com/). The goal of this repository is to provide an end-to-end experimentation and evaulation of Vision Transfomers on these types of systems.

Raising issues are encouraged so we know what features to prioritize. We want to inevitably work towards predicting regions of interest using masked patch prediction, so this is our top most priority on our road map.

## Documentation

### Installation

To install the package, you can simply run the following:

```console
pip install vit4elm
```

### Format Data from JSON to User Friendly images

To process the data, ensure you have a `data.json` from [statmechsims](https://www.statmechsims.com/). This json will contain all of your simulation data. Say you have the following directory:

```console
af_ising/
└── data.json
```

```python
# Import DataProcessor class
from vit4elm.data_processor import DataProcessor
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

However, if you want equidistant bins, you don't need to input `custom_intervals`, but rather `n_bins`.

```python
# Import DataProcessor class
from vit4elm.data_processor import DataProcessor
import numpy as np

# Instantiate necessary parameters.
json_path = 'af_ising/data.json'
data_dir = 'af_ising'

# Please ensure that the shape of custom_intervals is (n_bins, 2)
n_bins = 4

# Instantiate DataProcessor
processor = DataProcessor(json_path, data_dir, n_bins=n_bins)

# Let the processor process!
processor.process()
```

The result of either of the calls will lead to a directory with the following structure.

```console
af_ising/
├── csvs
│   └── data.csv
├── data
│   ├── bin0
│   ├── bin1
│   ├── bin2
│   └── bin3
├── data.json
└── experiments.json
```

The only difference between these is the boundaries that determine which image goes into which bin based off of the temperature. Again, this is assuming you want to bin into 4 bins. The `experiments.json` will be needed for instantiating a PyTorch dataset, so please do not delete it or alter anything in this directory.

## To Do

Smaller Stuff:

- [ ] Add notebooks for documentation
- [ ] Add cover image for repo

Bigger Stuff:

- [ ] Head wise visualizations
- [ ] Add masker to images