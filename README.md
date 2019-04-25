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

5. Optional: Download [pretrained weights](https://drive.google.com/uc?export=download&id=1eIvdtJ-d3TREpcpwXbCt3uG4sNnCanEK) 
(About 150 MB unzipped).
The weights have the following performances (These are slightly higher than the results in the paper because of averaging):
    - BBT0101: 99.55% WCP
    - BF0502: 98.85% WCP
    - ACCIO 36 clusters: 68.92% B-Cubed F-Score
    - ACCIO 40 clusters: 69.24% B-Cubed F-Score
Unzip the weights 



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