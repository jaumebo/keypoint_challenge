from audioop import reverse
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

SKELETON_GROUPS = {
    "right_arm": [3, 4, 5],
    "left_arm": [6, 7, 8],
    "right_leg": [9, 10, 11],
    "left_leg": [12, 13, 14],
}

CLASS_DICT = {"pick_up": 0, "put_back": 1, "raise_hand": 2, "standing": 3, "walking": 4}
ALL_CLASSES = ["pick_up", "put_back", "raise_hand", "standing", "walking"]
MAX_SEQUENCE_LENGTH = 300


def split_dataset(
    data_paths: List[Path],
    classes: List[str],
    test_size: int = 0.2,
    random_state: int = 123,
):
    train_paths = []
    val_paths = []
    for cl in classes:
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
    def __init__(
        self,
        dataset_dir: str,
        fill_value: dict = {"x": 0, "y": 0, "s": 0},
        classes: List[str] = ALL_CLASSES,
    ):

        self.class_dict = {cl: num for cl, num in zip(classes, range(len(classes)))}
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()
        self.fill_value = fill_value

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, classes=classes
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
        df["x"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
        df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
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

        return dataset, (self.max_sequence_length, 18, 3), self.class_dict


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
        classes: List[str] = ALL_CLASSES,
    ):

        self.class_dict = {cl: num for cl, num in zip(classes, range(len(classes)))}
        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()
        self.kept_positions = [
            i for i in range(1, 19) if KEYPOINT_DICT[i] not in skip_positions
        ]

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, classes=classes
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
        df["x"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
        df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
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

        return dataset, input_shape, self.class_dict


class SkeletonDataset(object):
    def __init__(
        self,
        dataset_dir: str,
        classes: List[str] = ALL_CLASSES,
        skeleton_groups: dict = SKELETON_GROUPS,
    ):

        self.class_dict = {cl: num for cl, num in zip(classes, range(len(classes)))}

        self.skeleton_groups = skeleton_groups
        self.all_positions = []
        for _, ids in self.skeleton_groups.items():
            self.all_positions += ids

        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, classes=classes
        )

    def __get_pose_dict(self, df: pd.DataFrame, frame_id: int, prev_pose: dict) -> dict:

        x_list = df[df["frame_id"] == frame_id]["x"].values
        y_list = df[df["frame_id"] == frame_id]["y"].values
        s_list = df[df["frame_id"] == frame_id]["score"].values
        keypoint_ids = list(df[df["frame_id"] == frame_id]["keypoint_id"].values)

        pose_dict = deepcopy(prev_pose)
        for keypoint, x, y, s in zip(keypoint_ids, x_list, y_list, s_list):
            if keypoint not in self.all_positions:
                continue
            pose_dict[keypoint] = {"x": x, "y": y, "s": s}

        return pose_dict

    def __get_input_sequence(self, df: pd.DataFrame) -> np.array:
        df["x"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
        df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
        sequence = []
        prev_pose = {i: {"x": 0.5, "y": 0.5, "s": 0} for i in self.all_positions}
        for frame_id in range(df["frame_id"].max() + 1):
            pose_dict = self.__get_pose_dict(df, frame_id, prev_pose)
            pose_array = []
            for group, keypoints in self.skeleton_groups.items():
                for keypoint in keypoints:
                    pose_array.append(
                        [
                            pose_dict[keypoint]["x"],
                            pose_dict[keypoint]["y"],
                            pose_dict[keypoint]["s"],
                        ]
                    )
            pose_array = np.array(pose_array)
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
                    shape=(None, len(self.all_positions), 3), dtype=tf.float32
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
            [self.max_sequence_length, len(self.all_positions), 3],
            [len(self.class_dict.keys())],
        )
        padding_values = (0.0, 0.0)

        dataset = dataset.padded_batch(batch_size, padded_shapes, padding_values)

        input_shape = (self.max_sequence_length, len(self.all_positions), 3)

        return dataset, input_shape, self.class_dict


class ReversedBinaryAugmentedDataset(object):
    def __init__(
        self,
        dataset_dir: str,
        classes: List[str] = ["pick_up", "put_back"],
        skeleton_groups: dict = SKELETON_GROUPS,
    ):

        self.class_dict = {cl: num for cl, num in zip(classes, range(len(classes)))}

        self.skeleton_groups = skeleton_groups
        self.all_positions = []
        for _, ids in self.skeleton_groups.items():
            self.all_positions += ids

        self.max_sequence_length = MAX_SEQUENCE_LENGTH
        self.dataset_dir = Path(dataset_dir).resolve()

        self.data_paths = list(self.dataset_dir.glob("**/*.csv"))

        self.train_paths, self.val_paths = split_dataset(
            self.data_paths, classes=classes
        )

    def __get_pose_dict(self, df: pd.DataFrame, frame_id: int, prev_pose: dict) -> dict:

        x_list = df[df["frame_id"] == frame_id]["x"].values
        y_list = df[df["frame_id"] == frame_id]["y"].values
        s_list = df[df["frame_id"] == frame_id]["score"].values
        keypoint_ids = list(df[df["frame_id"] == frame_id]["keypoint_id"].values)

        pose_dict = deepcopy(prev_pose)
        for keypoint, x, y, s in zip(keypoint_ids, x_list, y_list, s_list):
            if keypoint not in self.all_positions:
                continue
            pose_dict[keypoint] = {"x": x, "y": y, "s": s}

        return pose_dict

    def __get_input_sequence(self, df: pd.DataFrame, reverse: bool = False) -> np.array:
        df["x"] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min())
        df["y"] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
        sequence = []
        prev_pose = {i: {"x": 0.5, "y": 0.5, "s": 0} for i in self.all_positions}
        if reverse:
            frame_seq = range(df["frame_id"].max(), -1, -1)
        else:
            frame_seq = range(df["frame_id"].max() + 1)
        for frame_id in frame_seq:
            pose_dict = self.__get_pose_dict(df, frame_id, prev_pose)
            pose_array = []
            for _, keypoints in self.skeleton_groups.items():
                for keypoint in keypoints:
                    pose_array.append(
                        [
                            pose_dict[keypoint]["x"],
                            pose_dict[keypoint]["y"],
                            pose_dict[keypoint]["s"],
                        ]
                    )
            pose_array = np.array(pose_array)
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

            if (split == "train") and random.choice([True, False]):
                x = self.__get_input_sequence(df, reverse=True)
                y = 1 - get_class_key(csv_path, self.class_dict)
            else:
                x = self.__get_input_sequence(df, reverse=False)
                y = get_class_key(csv_path, self.class_dict)

            yield x, y

    def get_dataset(self, split: str, batch_size: int):

        dataset = tf.data.Dataset.from_generator(
            self.__data_generator,
            args=[split],
            output_signature=(
                tf.TensorSpec(
                    shape=(None, len(self.all_positions), 3), dtype=tf.float32
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
            [self.max_sequence_length, len(self.all_positions), 3],
            [len(self.class_dict.keys())],
        )
        padding_values = (0.0, 0.0)

        dataset = dataset.padded_batch(batch_size, padded_shapes, padding_values)

        input_shape = (self.max_sequence_length, len(self.all_positions), 3)

        return dataset, input_shape, self.class_dict
