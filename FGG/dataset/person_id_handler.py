from collections import Counter
from typing import List

from FGG.dataset.tracks import TrackCollection


class PersonIDHandler(object):
    unknown_key = "unknown"

    def __init__(self, main_characters: List[str], include_unknown=False):
        """
        Handles the overview over labels and person selection based on the selected characters.
        :param main_characters: The names of the selected characters.
        :param include_unknown: If True, all characters not in the main characters will be assigned a special
                                "unknown" class.
                                Otherwise they are excluded.
        """
        self.track_labels = []
        self.main_characters = main_characters
        self.include_unknown = include_unknown
        if include_unknown:
            self.main_characters.append(self.unknown_key)

    @property
    def num_characters(self):
        return len(self.main_characters)

    def __contains__(self, character):
        if character is None:
            # This happens when no labels are provided
            return True
        return self._map_name(character.lower()) in self.main_characters

    def get_distribution_of_persons(self, tracks: TrackCollection):
        labels = [track.label for track in tracks]
        counts = Counter(labels)
        labels = {key: counts.get(key) or 0 for key in self.main_characters}
        unknown_persons = [character for character in counts.keys() if character not in self.main_characters]
        unknowns = [counts.get(character) for character in unknown_persons]
        labels[self.unknown_key] = sum(unknowns)
        try:
            total_labels = sum(labels.values())
        except TypeError as e:
            raise ValueError("No labels in the validation set!") from e
        labels = {key: (val, val / total_labels) for key, val in labels.items()}
        return labels, {self.unknown_key: unknown_persons}

    def __getitem__(self, item: str):
        return self._map_name(item.lower())

    def _map_name(self, name_lower_case: str):
        return name_lower_case if name_lower_case in self.main_characters else self.unknown_key