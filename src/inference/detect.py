import argparse
from copy import deepcopy
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


class Detector(object):
    def __init__(self, primary_model_path: str, secondary_model_path: str):

        self.primary_model = load_model(primary_model_path)
        self.secondary_model = load_model(secondary_model_path)

        self.class_dict = {
            0: "pick_up",
            1: "put_back",
            2: "raise_hand",
            3: "standing",
            4: "walking",
        }

        self.skeleton_groups = {
            "right_arm": [3, 4, 5],
            "left_arm": [6, 7, 8],
            "right_leg": [9, 10, 11],
            "left_leg": [12, 13, 14],
        }
        self.all_positions = []
        for _, ids in self.skeleton_groups.items():
            self.all_positions += ids

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

        inputs = np.expand_dims(np.array(sequence), axis=0)

        inputs = np.pad(
            inputs, ((0, 0), (0, 300 - inputs.shape[1]), (0, 0), (0, 0)), "constant"
        )

        return inputs

    def __process_csv(self, csv_path: Path) -> np.array:

        df = pd.read_csv(
            csv_path,
            header=None,
            names=["frame_id", "keypoint_id", "x", "y", "score"],
        )

        inputs = self.__get_input_sequence(df)

        return inputs

    def predict(self, csv_paths: List[Path]):

        results = []

        for path in csv_paths:

            try:

                assert path.is_file(), f"CSV file does not exist: {path}"

                inputs = self.__process_csv(path)

                res = self.primary_model.predict(inputs)
                class_id = tf.argmax(res, axis=-1)[0].numpy()
                if class_id < 2:
                    res = self.secondary_model.predict(inputs)
                    class_id = tf.argmax(res, axis=-1)[0].numpy()

                results.append((path.name, class_id, self.class_dict[int(class_id)]))

            except:
                print(f"FAILED WITH CSV: {path.name}")

        return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--primary_model_path",
        type=str,
        nargs="?",
        default="/Users/jaumebrossa/Code/AI/v7_challenge/keypoint_challenge/saved_models/primary_model",
    )
    parser.add_argument(
        "--secondary_model_path",
        type=str,
        nargs="?",
        default="/Users/jaumebrossa/Code/AI/v7_challenge/keypoint_challenge/saved_models/secondary_model",
    )
    args = parser.parse_args()

    detector = Detector(args.primary_model_path, args.secondary_model_path)

    csv_paths = Path("/Users/jaumebrossa/Code/AI/v7_challenge/data/test").glob(
        "**/*.csv"
    )

    results = detector.predict(csv_paths)

    df = pd.DataFrame(columns = ["file","class"])

    for res in results:
        df = df.append({"file": res[0], "class": res[2]})

    df.to_csv("test_results.csv")

