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
4. Get input datasets (from vivek.sharma@kit.edu?).
Put them inside this repository in a folder named "data".
There should be a subfolder for each dataset named after it, e.g. "bbt".
Inside each of them there should be a subfolder for each episode, e.g. "bbt/bbt0101/tracks.mat".
Alternatively you can change the paths in `FGG/dataset/input_data.py` within each dataset.


5. Optional: Download [pretrained weights](https://drive.google.com/uc?export=download&id=1VkFMpiMkYI1_SpBQ05EQjB0Y2P7vYdgr) 
    (About 650 MB unzipped). Unzip them inside the repository so they will be automatically found.
    The weights have the following performances (These are the best models that we found) 
    - BBT0101: 99.78% WCP
    - BBT0102: 99.72% WCP
    - BBT0103: 99.64% WCP
    - BBT0104: 99.27% WCP
    - BBT0105: 98.80% WCP
    - BBT0106: 93.22% WCP
    - BF0501: 97.92% WCP
    - BF0502: 98.85% WCP
    - BF0503: 98.71% WCP
    - BF0504: 96.83% WCP
    - BF0505: 97.44% WCP
    - BF0506: 97.53% WCP 
    - ACCIO 36 clusters: 68.92% B-Cubed F-Score
    - ACCIO 40 clusters: 69.24% B-Cubed F-Score
    

## Features

You can download the FGG features for the main characters of [BF](https://drive.google.com/uc?export=download&id=1QAjeSEAEMP4vgmbj42ypFPYwqK_lxP1n), [BBT](https://drive.google.com/uc?export=download&id=1zG6J_cUsIZGm8DVqlPYo6e9sGnlHMwnf) and [Accio](https://drive.google.com/uc?export=download&id=1yhtipY3l1U-geSSqk2ikgIg0M6ZmrMEK).
These are the output features for the weights above.
Please keep in mind that the output dimension is not the same as the input dimension (i.e. there are less FGG features)
due to the sub-track representation.

Additionally, you can download another set of features for the main characters([BF](https://drive.google.com/uc?export=download&id=1Lok3O2oZhA5zdLeivpccROZREU9fAiGW),[BBT](https://drive.google.com/uc?export=download&id=1p_6kJCWMtANSk_LnHgFACjV6TESIfBr8)).
These are trained so that there is one feature for every 10 frames in each track, e.g. a track with 27 faces results in 3 features.
Their clustering performance is:
- BBT0101: 99.50% WCP
- BBT0102: 98.16% WCP
- BBT0103: 99.42% WCP
- BBT0104: 99.02% WCP
- BBT0105: 99.26% WCP
- BBT0106: 93.44% WCP
- Bf0501: 96.67% WCP
- Bf0502: 97.01% WCP
- Bf0503: 97.96% WCP
- Bf0504: 91.59% WCP
- Bf0505: 96.44% WCP
- Bf0506: 96.51% WCP


__Output format__:
The files are of type HDF5 and follow the following format:
You can read them with h5py.
- Dataset "FGG_features": These are the output features of shape `|V|x128`.
V is the number of nodes in the graph and depends on the input size track length.
We split tracks into sub-tracks once they reach a certain length,
so the output contains one or more features for each track (temporally ordered).
For example on Buffy Episode 6, `V=3176` for `535` full tracks.
Please see the paper Section 3.2 §Split Variants for details.
- Group "Tracks": Contains the following  datasets. Each dataset has `|V|` rows.
    - "tracker_id": The id of the tracker to which the features belong.
    - "label": The ground-truth label. If you run  FGG on a dataset where there are no labels, this will be missing in the output file.
    - "start_frame": start frame of this sub-track
    - "end_frame": end frame of this sub-track
    - "subtrack_id": counter of sub-tracks to the corresponding "tracker_id"
    - "predicted_label": The predicted label. If you run  FGG on a dataset where there are no labels, this will contain the cluster assignments.
    
    
## Instructions


__Expected input format__ :
We expect the inputs to be in a HDF5 file.
The following structure is assumed:

- Dataset "features_vgg": Contains the features. Should be of shape `2048xFxC` where C is the number of crops (i.e. for 10-crop C=10) and F us the number of faces. C is optional.
- Group "Tracks": Contains the following groups. Each group contains a dataset with the content for each track.
    - "trackerId": The id of the tracker to which the features belong.
    - "label": The ground-truth label.
    - "numframes": Number of frames in the track.
    - "frames": The frame numbers of the track.


### General

The code is organized into separate components.
The input data is represented by an `InputData` class.
If you want to run on your own dataset, you need to subclass `InputData`.
You can change the expected names for the HDF5 datasets easily if required, see
`class::FGG/dataset/input_data.py.Accio` for an example.

This has been done for BBT, BF and ACCIO. 
You can use those as an example.
The code is also documented.
By instantiating this class you define the main characters, number of clusters and which episodes to train/test/validate on.

Running FGG is done via the `Experiment` class.
Each experiment is named by the name of the class and a timestamp when the experiment is started.
Here you can specify which dataset to use and much more.
The exact things you can change are all contained in `FGG/config.py`.
You also have the option to run multiple experiments in serial, but without having to restart anything.
You can provide separate configurations for each experiment.
                          
To see most metrics please start a visdom server in a separate terminal (just type `visdom`).
Otherwise only WCP will be reported and you will see a HTTP connection error (you can ignore that, it will work anyways).                         
                       
### Training

The base `Experiment` class is used for training.
You can find it in `train.py`.
To train please subclass `Experiment` and change whatever you need.
For each dataset you should at least specify `episode_index_train`.
By default an evaluation step will be performed on the same episode when training is done.

The default setting is to train on BBT0101, call it with
```
python train.py
```
There also exists a predefined experiment type for the other datasets.

The following output files will be created:

- In `experiment_results/` you will find a file `[Experiment name].csv` listing all the outputs, and any changes of the model you might have recorded.
- in `runs/` each run within the experiment will have their own sub-folder. The folders are separated by dataset name.
  Inside you will find a folder for each run: `[Experiment name]_[run_idx]`. This folder contains:
  - features.h5: The output features. Format described above.
  - checkpoint.tar: The checkpoint containing parameters of weights and optimizer.
  - run_info.json: Lists the full configuration of that run. Nice if you're not sure what you did for an experiment.
  - statistics.h5: Saves the metrics of the run. Please refer to `FGG/persistence/statistics.py` for the format.
  - If you had a visdom server running, you will also receive replay_log.json with which you can recreate the visdom plots.
  
### Evaluation & Feature Extraction

The `EvalExperiment` subclass is provided for convenience to allow to only evaluate a pre-trained model
without any further training.
There are several evaluation experiments implemented in `evaluate.py`.
Only the experiment csv, features.h5, run_info.json and possibly the replay_log will be saved.

The default setting is to evaluate on all episodes of BBT, call it with
```
python evaluate.py
```

If you want to produce the output features with one feature per 10 tracks,
change the exeriment type in `evaluate.py`.
Please also see the comments in the code.

### Inference

In case you do not have any labels for you dataset, you can use `infer.py`.
This is similar to evaluation, but we disable any scoring.
We provide an example with BBT on how to turn off the loading of the labels.
Only the features will be saved.
    
    
    
### Running without Experiments

If you want to run FGG without experiments you need to instantiate a config and a runner:
```python

from FGG.config import FaceGroupingConfig
from FGG.runner import Runner
from FGG.persistence.run_configuration import enable_auto_run_save

enable_auto_run_save()
config = FaceGroupingConfig()  # You can modify this to change the mode's behavior.
experiment = Runner.from_config(config)
experiment.train()
```
## B-CUBED score Implementation

The code for the B-Cubed metric is taken from [this repository](https://github.com/m-wiesner/BCUBED).
It is distributed with the Apache 2.0 license.
There is no pip package for it currently, so we include it here as well.


# Citation

If you find this work useful, please cite

```
TODO
```
