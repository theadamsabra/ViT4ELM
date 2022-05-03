# Applications of Vision Transformers for Exotic Lattice Models 

## About
Applications of Vision Transformers to Exotic Lattice Models. This repository contains all of the finetuned models as well as core scripts to reproduce results and experiment on new data. The experimentation suite that this repository builds on top of can be found [in this website](https://www.statmechsims.com/). The goal of this repository is to provide an end-to-end experimentation and evaulation of Vision Transfomers on these types of systems.

Raising issues are encouraged so we know what features to prioritize. We want to inevitably work towards predicting regions of interest using masked patch prediction, so this is our top most priority on our road map. 

# Documentation:

## Format Data from JSON to User Friendly images.
To process the data, ensure you have a `data.json` from [statmechsims](https://www.statmechsims.com/). This json will contain all of your simulation data.

The result of the above will lead to a new directory named `af_ising` with the following structure:

```console
python3 data_handle.py --json_path="/path/to/data.json" --n_bins=4 --data_dir"af_ising"
```

The result of the following CLI call will lead to a directory with the following structure. Binning is set by the temperature value, all of equal length between the minimum and maximum temperature.

```console
af_ising/
├── bin0 
├── bin1
├── bin2 
└── bin3 
```

## Training the Model

To train on the pretrained ViT Base 32 model with default configurations for feature extraction, run the following command.

```console
python3 trainer.py --data_dir="af_ising" --test_split=0.4
```
Make sure that the `data_dir` parameter from both `data_handle` and `trainer` CLI calls are the same and there should be no issue.

# To Do:

Low Hanging Fruit:
- [ ] Add nice cover image to repo
- [ ] Add evaluation script
   - [ ] Load model
   - [ ] Run on validation set (need to save from trainer)
   - [ ] Add train/test/validation + saving validation set mechanism
   - [ ] Get confusion matrix

Bigger Stuff:
- [ ] Head wise visualizations
- [ ] Add masker to images
