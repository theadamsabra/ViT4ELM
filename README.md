# Applications of Vision Transformers for Exotic Lattice Models

## About

Applications of Vision Transformers to Exotic Lattice Models. This repository contains all of the finetuned models as well as core scripts to reproduce results and experiment on new data. The experimentation suite that this repository builds on top of can be found [in this website](https://www.statmechsims.com/). The goal of this repository is to provide an end-to-end experimentation and evaulation of Vision Transfomers on these types of systems.

Raising issues are encouraged so we know what features to prioritize. We want to inevitably work towards predicting regions of interest using masked patch prediction, so this is our top most priority on our road map.

The main goal of building our own framework (PyTorch datasets, Trainer, etc.) is to further expand upon the ideas of implementing vision transformers in these systems. This includes studying saliency and attention maps of CNN and Transformer based architectures in attempts of a deeper understanding of how these neural network systems may learn from the underlying physics of exotic lattice models.

## Documentation

### Installation

To install the package, you can simply run the following:

```console
pip install vit4elm
```

### Notebooks:

For full-fledged documentation on the various classes that have been created within this repository, there are notebooks to view for your discretion.


## To Do

Bigger Stuff:

- [ ] Head wise visualizations
- [ ] Add masker to images