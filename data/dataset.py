from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf

CLASS_DICT = {"pick_up": 0, "put_back": 1, "raise_hand": 2, "standing": 3, "walking": 4}


class SequenceDatasetCreator(object):
    def __init__(self, dataset_dir: str, fill_value: dict = {"x": 0, "y": 0, "s": 0}):

        self.class_dict = CLASS_DICT
        self.dataset_dir = Path(dataset_dir).resolve()
        self.fill_value = fill_value

        self.data_paths_gen = self.dataset_dir.glob("**/*.csv")

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

    def __get_class(self, csv_path):
        name_list = csv_path.name.split(".")[0].split("_")[1:]
        name = "_".join(name_list)
        class_key = self.class_dict[name]
        return class_key

    def __data_generator(self) -> Callable:

        for csv_path in self.data_paths_gen:
            df = pd.read_csv(
                csv_path,
                header=None,
                names=["frame_id", "keypoint_id", "x", "y", "score"],
            )

            x = self.__get_input_sequence(df)
            y = self.__get_class(csv_path)

            yield x, y

    def get_dataset(self):

        dataset = tf.data.Dataset.from_generator(
            self.__data_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, 18, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
            ),
        )

        # Label one-hot encoding
        dataset = dataset.map(
            lambda x, y: (x, tf.one_hot(y, len(self.class_dict.keys())))
        )

        return dataset


if __name__ == "__main__":

    dataset_creator = SequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/"
    )

    dataset = dataset_creator.get_dataset()

    for (x, y) in dataset:
        print(x)
        print(y)
        pass
