# Self-supervised Face Grouping on Graphs

## Installation

1. Clone this repository:
```
git clone git@github.com:RunOrVeith/FGG.git
```
2. Recommended: Create a python virtual environment:
```
python -m venv vfgg
source vfgg/bin/activate
```
3. Install the dependencies:
```
pip install -r FGG/requirements.txt
```
4. Optional: Download the datasets

todo

5. Optional: Download pretrained weights

todo

## B-CUBED Implementation

The code for the B-Cubed metric is taken from [this repository](https://github.com/m-wiesner/BCUBED).
It is distributed with the Apache 2.0 license.
There is no pip package for it currently, so we include it here as well.

## TODO


- [ ] Inference without known labels
- [ ] Docstrings
- [ ] Convert input data
- [ ] Document output format
- [ ] Store only the 29th epoch of checkpoints

# Citation

If you find this work useful, please cite

```
TODO
```

# License