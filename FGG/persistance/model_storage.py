from pathlib import Path
import copy

import torch
import numpy as np


class ModelStorage(object):

    def __init__(self, path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 load_from_old_path: str = None, strict=True):
        load_path = load_from_old_path or path
        self.model = model
        self.optimizer = optimizer
        if Path(load_path).is_file():
            print(f"Setting up checkpoint from {load_path}.")
            try:
                self.checkpoint = torch.load(load_path)
            except RuntimeError:
                print("Runtime Error detected, trying to remap checkpoint to CPU...")
                self.checkpoint = torch.load(load_path, map_location="cpu")
            self.load_path = load_path
        else:
            self.checkpoint = {}
            self.load_path = None
        self.path = path
        self.strict = strict
        self._current_count = 0
        try:
            self._current_count = max(count for count, performance in self.checkpoint.keys())
        except ValueError:
            pass

    def next_epoch(self):
        self._current_count += 1

    def save(self, performance_indicator=None):
        current_state = dict(
            model_state_dict=copy.deepcopy(self.model.state_dict()),
            optimizer_state_dict=copy.deepcopy(self.optimizer.state_dict()),
        )
        torch.save({(self._current_count, performance_indicator): current_state}, str(Path(self.path)))
        print("saved model with key", self._current_count, performance_indicator)

    def load_most_recent(self):
        matches = self._matching_checkpoint_keys(lambda c, p: c == self._current_count)
        if len(matches) > 0:
            key = matches[0]
            print("Loading most recent model.")
            return self._load(key=key)
        else:
            print("No model found in checkpoint. Starting from scratch.")
            return self.model, self.optimizer

    def load_best(self, determine=np.argmax):
        matches = self._matching_checkpoint_keys(lambda c, p: p is not None)
        if len(matches) > 0:
            counts, performances = zip(*matches)
            index = determine(performances)
            key = matches[np.asscalar(index)]
            print("Loading best model")
            return self._load(key=key)
        else:
            print("No checkpoint with performance key found. Starting model from scratch.")
            return self.model, self.optimizer

    def _matching_checkpoint_keys(self, use_check):
        matches = [(count, performance) for count, performance in self.checkpoint.keys() if
                   use_check(count, performance)]
        return matches

    def load_from_epoch(self, epoch):
        matches = self._matching_checkpoint_keys(lambda c, p: c == epoch)
        if len(matches) > 0:
            key = matches[0]
            print(f"Loading model from epoch {epoch}")
            return self._load(key=key)
        else:
            print(f"Epoch {epoch} not found in checkpoint. Starting model from scratch.")
            return self.model, self.optimizer

    def _load(self, key):
        print(f"Loading model from {self.load_path} with key {key}")
        self.model.load_state_dict(self.checkpoint[key]["model_state_dict"], strict=self.strict)
        self.optimizer.load_state_dict(self.checkpoint[key]["optimizer_state_dict"])
        return self.model, self.optimizer
