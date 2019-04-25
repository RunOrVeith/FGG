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
4. Get input datasets (from vivek.sharma@kit.edu?)

5. Optional: Download [pretrained weights](https://drive.google.com/uc?export=download&id=120LTHw4Hcdvp7HrpWpn1ptl9bRv1H8dz) 
    (About 150 MB unzipped). Unzip them inside the repository so they will be automatically found.
    The weights have the following performances
    - BF0502: 98.85% WCP
    - ACCIO 36 clusters: 68.92% B-Cubed F-Score
    - ACCIO 40 clusters: 69.24% B-Cubed F-Score
    

## Features
You can download the FGG features for BBT0101, BF0502 and Accio [here](https://drive.google.com/uc?export=download&id=1_vZ4TpHGVhD6Im-T6GeDr7SzC_-E9quY).
Please keep in mind that the output dimension is not the same as the input dimension (i.e. there are less FGG features)
due to the sub-track representation.

The files are of type HDF5 and follow the following format:

- Dataset "FGG_features": These are the output features of shape `|V|x128`.
- Group "Tracks": Contains the following  datasets. Each dataset has `|V|` rows.
    - "tracker_id": The id of the tracker to which the features belong.
    - "label": The ground-truth label. If you run  FGG on a dataset where there are no labels, this will be missing in the output file.
    - "start_frame": start frame of this sub-track
    - "end_frame": end frame of this sub-track
    - "subtrack_id": counter of sub-tracks to the corresponding "tracker_id"
    - "predicted_label": The predicted label. If you run  FGG on a dataset where there are no labels, this will contain the cluster assignments.
    
    
## Instructions

To obtain the input data matrices for BBT, BF or ACCIO please contact vivek.sharma@kit.edu.
Put them inside this repository in a folder named "data".
There should be a subfolder for each dataset named "bf", "bbt" and "accio".
Inside each of them there should be a subfolder for each episode, e.g. bbt/bbt0101/tracks.mat (except in accio).
Alternatively you can change the paths in `FGG/dataset/input_data.py`.


__Expected input__:
We expect the inputs to be in a HDF5 file.
The following structure is assumed:

- Dataset "features_vgg": Contains the features. Should be of shape `CxFx2048` where C is the number of crops (i.e. for 10-crop C=10) and F us the number of faces. C is optional.
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
`FGG/dataset/input_data.py.Accio` for an example.

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

The following output files will be created:

- In `experiment_results/` you will find a file `[Experiment name].csv` listing all the outputs, and any changes of the model you might have recorded.
- in `runs/` each run within the experiment will have their own sub-folder. The folders are separated by dataset name.
  Inside you will find a folder for each run: `[Experiment name]_[run_idx]`. This folder contains:
  - features.h5: The output features. Format described above.
  - checkpoint.tar: The checkpoint containing parameters of weights and optimizer.
  - run_info.json: Lists the full configuration of that run. Nice if you're not sure what you did for an experiment.
  - statistics.h5: Saves the metrics of the run. Please refer to `FGG/persistence/statistics.py` for the format.
  - If you had a visdom server running, you will also receive replay_log.json with which you can recreate the visdom plots.
  
### Evaluation

The `EvalExperiment` subclass is provided for convenience to allow to only evaluate a pre-trained model
without any further training.
There are several evaluation experiments implemented in `evaluate.py`.
Only the experiment csv, features.h5, run_info.json and possibly the replay_log will be saved.

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

# License