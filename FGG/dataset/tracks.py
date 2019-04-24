from typing import List, Union

import h5py
import numpy as np
import pandas as pd


class TrackCollection(object):

    @staticmethod
    def from_hdf5_track_file(track_file, person_id_handler: "PersonIDHandler", crop_idx=4,
                             tracks_header="Tracks", features_header="features_vgg",
                             label_header="label", num_frames_header="numframes",
                             tracker_id_header="trackerId", frames_header="frames"):

        def read_str(tracks, ref):
            label = tracks[ref]
            label = "".join(chr(x) for x in label)
            return label

        def extract_column(tracks, name, dtype=str):
            # This is some very ugly HDF5 parsing code
            col = tracks[name]
            shape = col.shape
            result = []
            for row in range(shape[0]):
                if dtype == str:
                    try:
                        result.append(read_str(tracks, tracks[col[row][0]][0][0]))
                    except AttributeError:
                        result.append(read_str(tracks, col[row][0]))
                else:
                    result.append(tracks[col[row][0]][()].astype(dtype))
            try:
                return np.asarray(result).squeeze()
            except ValueError:
                return result

        f_tracks = h5py.File(track_file, "r")
        tracks = f_tracks.get(tracks_header)

        track_list = []
        frames = extract_column(tracks, frames_header, dtype=np.uint32)
        start_frames, end_frames = zip(*map(lambda fs: (fs.min(), fs.max()), frames))

        data = pd.DataFrame({"start_frame": start_frames,
                             "end_frame": end_frames,
                             "numframes": extract_column(tracks, num_frames_header, np.uint32),
                             "tracker_id": extract_column(tracks, tracker_id_header, np.uint32)
                             })
        if label_header is not None:
            data["label"] = extract_column(tracks, label_header)
        else:
            data["label"] = None
        i = 0
        print("Loading VGG2 features, this may take a while...")
        features = f_tracks.get(features_header)
        if len(features.shape) == 2:
            assert crop_idx == 0
            features = features[...].astype(np.float32)
        else:
            assert len(features.shape) == 3
            features = features[crop_idx].astype(np.float32)

        required_features = []
        required_feature_counter = 0
        for idx, track in data.iterrows():
            track_obj = Track(tracker_id=track["tracker_id"], label=track["label"],
                              start_frame=track["start_frame"], end_frame=track["end_frame"],
                              feat_indices=list(
                                  range(required_feature_counter, track["numframes"] + required_feature_counter)
                              ),
                              )
            if track.label in person_id_handler:
                feat_indices = list(range(i, track["numframes"] + i))
                assert len(feat_indices) == len(track_obj) == len(track_obj.feat_indices)
                required_features.append(features[feat_indices])
                required_feature_counter += len(feat_indices)
                track_list.append(track_obj)
            i += track["numframes"]

        f_tracks.close()
        required_features = np.concatenate(required_features)
        assert sum(len(track) for track in track_list) == len(required_features) == sum(
            len(track.feat_indices) for track in track_list)
        assert len(set([track.tracker_id for track in track_list])) == len(track_list)
        print(f"Number of individual faces: {required_features.shape}")
        return TrackCollection(tracks=track_list, features=required_features,
                               person_id_handler=person_id_handler)

    def __init__(self, tracks: List["Track"], features: np.ndarray, person_id_handler: "PersonIDHandler"):
        self.tracks = tracks
        self.last_frame_with_track = max(track.end_frame for track in self)
        self.features = features
        self.person_id_handler = person_id_handler
        self.feature_embedding = None

    def __getitem__(self, track: "Track") -> np.ndarray:
        return self.features[track.feat_indices]

    def tracks_in_frame(self, frame: int):
        return [track for track in self.tracks if track.contains_frame(frame=frame)]

    def __iter__(self):
        yield from self.tracks

    def pooled_features(self):
        pooled_features = np.array([self[track].mean(axis=0) for track in sorted(self)])
        assert len(pooled_features) == len(self.tracks)
        return pooled_features

    def set_prediction_output(self, feature_embedding, predicted_labels):
        assert feature_embedding.shape[0] == len(self.tracks) == len(predicted_labels)
        self.feature_embedding = feature_embedding
        for track, prediction in zip(self, predicted_labels):
            track.predicted_label = self.person_id_handler.main_characters[prediction]

    def output_features(self, output_file):
        if self.feature_embedding is None:
            raise RuntimeError("Can not output feature embedding: not computed.")
        with h5py.File(output_file, "w") as f:
            f.create_dataset("FGG_features", data=self.feature_embedding)
            info = f.create_group("Tracks")
            tracker_id, label, start_frame, end_frame, predicted_label, subtrack_id = zip(
                *[track.h5_info() for track in self.tracks])

            info.create_dataset("tracker_id", data=np.array(tracker_id))
            if not all(x is None for x in label):
                info.create_dataset("label", data=np.array(label))
            info.create_dataset("start_frame", data=np.array(start_frame))
            info.create_dataset("end_frame", data=np.array(end_frame))
            info.create_dataset("subtrack_id", data=np.array(subtrack_id))
            info.create_dataset("predicted_label", data=np.array(predicted_label))


class Track(object):

    def __init__(self, tracker_id, label, start_frame, end_frame,
                 feat_indices, subtrack_id=None):
        self.tracker_id = tracker_id
        self.label = label
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.feat_indices = feat_indices
        self.predicted_label = None
        self.subtrack_id = subtrack_id
        assert len(self) == len(self.feat_indices), f"{str(self)}, {len(self)}, {len(self.feat_indices)}"

    def h5_info(self):

        try:
            label = self.label.encode("utf-8")
        except AttributeError:
            label = None
        return self.tracker_id, label, self.start_frame, self.end_frame, self.predicted_label.encode("utf-8"), self.subtrack_id

    def __str__(self):
        return f"Track {self.tracker_id}({self.subtrack_id if self.subtrack_id is not None else ''}) - {self.label}({self.predicted_label or ''}):" \
            f" frames [{self.start_frame},{self.end_frame}]" \
            f" feature rows [{min(self.feat_indices)}, {max(self.feat_indices)}]"

    def __lt__(self, other):
        # Make tracks sortable by tracker ID so we can iterate over the graph notes in a fixed order
        if self.subtrack_id is None or other.subtrack_id is None:
            return self.tracker_id < other.tracker_id
        else:
            return self.tracker_id < other.tracker_id and self.subtrack_id < other.subtrack_id

    @property
    def frame_range(self):
        return range(self.start_frame, self.end_frame + 1)

    def contains_frame(self, frame):
        return frame in self.frame_range

    def __len__(self):
        return len(list(self.frame_range))

    def overlaps(self, other):
        return len(set(self.frame_range).intersection(other.frame_range)) > 0

    def split(self, into: Union[int, np.ndarray]):
        if isinstance(into, int):
            assert 1 <= into <= len(self), f"{into} : {len(self)}"
        else:
            assert len(into) == 0 or into.max() <= len(self)

        subtracks = []
        sub_indices = np.array([list(self.frame_range), self.feat_indices])
        for subtrack_id, indices in enumerate(np.array_split(sub_indices, into, axis=-1)):
            frames, feat_indices = indices
            subtrack = Track(tracker_id=self.tracker_id, label=self.label, subtrack_id=subtrack_id,
                             start_frame=frames.min(), end_frame=frames.max(),
                             feat_indices=feat_indices)
            # feat_indices=self.feat_indices)
            subtracks.append(subtrack)
        # warnings.warn("Using copied over full features over the graph!")
        if isinstance(into, int):
            assert len(subtracks) == into
        else:
            assert len(subtracks) == len(into) + 1
        return subtracks
