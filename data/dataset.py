import random
from copy import deepcopy
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

KEYPOINT_DICT = {
    1: "nose",
    2: "neck",
    3: "right_shoulder",
    4: "right_elbow",
    5: "right_wrist",
    6: "left_shoulder",
    7: "left_elbow",
    8: "left_wrist",
    9: "right_hip",
    10: "right_knee",
    11: "right_ankle",
    12: "left_hip",
    13: "left_knee",
    14: "left_ankle",
    15: "right_eye",
    16: "left_eye",
    17: "right_ear",
    18: "left_ear",
}

CLASS_DICT = {"pick_up": 0, "put_back": 1, "raise_hand": 2, "standing": 3, "walking": 4}
MAX_SEQUENCE_LENGTH = 170


def split_dataset(
    data_paths: List[Path],
    class_dict: dict,
    test_size: int = 0.2,
    random_state: int = 123,
):
    train_paths = []
    val_paths = []
    for cl in class_dict.keys():
        cl_paths = [path for path in data_paths if cl in str(path)]
        train_class_paths, val_class_paths = train_test_split(
            cl_paths, test_size=test_size, random_state=random_state
        )
        train_paths += train_class_paths
        val_paths += val_class_paths
    random.shuffle(train_paths)
    random.shuffle(val_paths)
    return train_paths, val_paths


def get_class_key(csv_path: Path, class_dict: dict):
    name_list = csv_path.name.split(".")[0].split("_")[1:]
    name = "_".join(name_list)
    class_key = class_dict[name]
    return class_key


class SimpleSequenceDatasetCreator(object):
    def __init__(self, dataset_dir: str, fill_value: dict = {"x": 0, "y": 0, "s": 0}):

        self.class_dict = CLASS_DICT
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()
        self.fill_value = fill_value

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, self.class_dict
        )

    def __get_pose_dict(self, df: pd.DataFrame, frame_id: int) -> dict:

        x_list = df[df["frame_id"] == frame_id]["x"].values
        y_list = df[df["frame_id"] == frame_id]["y"].values
        s_list = df[df["frame_id"] == frame_id]["score"].values
        keypoint_ids = list(df[df["frame_id"] == frame_id]["keypoint_id"].values)

        pose_dict = {i: self.fill_value for i in range(1, 19)}
        for keypoint, x, y, s in zip(keypoint_ids, x_list, y_list, s_list):
            pose_dict[keypoint] = {"x": x, "y": y, "s": s}

        return pose_dict

    def __get_input_sequence(self, df: pd.DataFrame) -> np.array:
        sequence = []
        for frame_id in range(df["frame_id"].max() + 1):
            pose_dict = self.__get_pose_dict(df, frame_id)
            pose_array = np.array(
                [
                    [pose_dict[i]["x"], pose_dict[i]["y"], pose_dict[i]["s"]]
                    for i in range(1, 19)
                ]
            )
            sequence.append(pose_array)

        sequence = np.array(sequence)

        return sequence

    def __data_generator(self, split: str) -> Callable:

        split = split.decode("utf-8")

        if split == "train":
            path_list = self.train_paths
        elif split == "val":
            path_list = self.val_paths
        else:
            raise ValueError(
                f"Dataset split {str(split)} not supported: only train and val available"
            )

        for csv_path in path_list:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["frame_id", "keypoint_id", "x", "y", "score"],
            )

            x = self.__get_input_sequence(df)
            y = get_class_key(csv_path, self.class_dict)

            yield x, y

    def get_dataset(self, split: str, batch_size: int):

        dataset = tf.data.Dataset.from_generator(
            self.__data_generator,
            args=[split],
            output_signature=(
                tf.TensorSpec(shape=(None, 18, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        # Label one-hot encoding
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, len(self.class_dict.keys())))
        )

        # Shuffle dataset
        if split == "train":
            dataset = dataset.shuffle(50)

        # Batch dataset
        padded_shapes = (
            [self.max_sequence_length, 18, 3],
            [len(self.class_dict.keys())],
        )
        padding_values = (0.0, 0.0)

        dataset = dataset.padded_batch(batch_size, padded_shapes, padding_values)

        return dataset


class ProcessedSequenceDatasetCreator(object):
    def __init__(
        self,
        dataset_dir: str,
        skip_positions: List[str] = [
            "nose",
            "neck",
            "right_ear",
            "left_ear",
            "right_eye",
            "right_eye",
        ],
    ):

        self.class_dict = CLASS_DICT
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()
        self.kept_positions = [
            i for i in range(1, 19) if KEYPOINT_DICT[i] not in skip_positions
        ]

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, self.class_dict
        )

    def __get_pose_dict(self, df: pd.DataFrame, frame_id: int, prev_pose: dict) -> dict:

        x_list = df[df["frame_id"] == frame_id]["x"].values
        y_list = df[df["frame_id"] == frame_id]["y"].values
        s_list = df[df["frame_id"] == frame_id]["score"].values
        keypoint_ids = list(df[df["frame_id"] == frame_id]["keypoint_id"].values)

        pose_dict = deepcopy(prev_pose)
        for keypoint, x, y, s in zip(keypoint_ids, x_list, y_list, s_list):
            if keypoint not in self.kept_positions:
                continue
            pose_dict[keypoint] = {"x": x, "y": y, "s": s}

        return pose_dict

    def __get_input_sequence(self, df: pd.DataFrame) -> np.array:
        sequence = []
        prev_pose = {i: {"x": 0.5, "y": 0.5, "s": 0} for i in self.kept_positions}
        for frame_id in range(df["frame_id"].max() + 1):
            pose_dict = self.__get_pose_dict(df, frame_id, prev_pose)
            prev_pose = deepcopy(prev_pose)
            pose_array = np.array(
                [
                    [pose_dict[i]["x"], pose_dict[i]["y"], pose_dict[i]["s"]]
                    for i in self.kept_positions
                ]
            )
            sequence.append(pose_array)

        sequence = np.array(sequence)

        return sequence

    def __data_generator(self, split: str) -> Callable:

        split = split.decode("utf-8")

        if split == "train":
            path_list = self.train_paths
        elif split == "val":
            path_list = self.val_paths
        else:
            raise ValueError(
                f"Dataset split {str(split)} not supported: only train and val available"
            )

        for csv_path in path_list:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["frame_id", "keypoint_id", "x", "y", "score"],
            )

            x = self.__get_input_sequence(df)
            y = get_class_key(csv_path, self.class_dict)

            yield x, y

    def get_dataset(self, split: str, batch_size: int):

        dataset = tf.data.Dataset.from_generator(
            self.__data_generator,
            args=[split],
            output_signature=(
                tf.TensorSpec(
                    shape=(None, len(self.kept_positions), 3), dtype=tf.float32
                ),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        # Label one-hot encoding
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, len(self.class_dict.keys())))
        )

        # Shuffle dataset
        if split == "train":
            dataset = dataset.shuffle(50)

        # Batch dataset
        padded_shapes = (
            [self.max_sequence_length, len(self.kept_positions), 3],
            [len(self.class_dict.keys())],
        )
        padding_values = (0.0, 0.0)

        dataset = dataset.padded_batch(batch_size, padded_shapes, padding_values)

        input_shape = (self.max_sequence_length, len(self.kept_positions), 3)

        return dataset, input_shape


if __name__ == "__main__":

    dataset_creator = ProcessedSequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/"
    )

    dataset, input_shape = dataset_creator.get_dataset("val", 4)

    y_true = []
    for x, y in dataset:
        y_true += tf.argmax(y, axis=-1).numpy().tolist()
        pass

    print(y_true)
