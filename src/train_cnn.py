import argparse

import wandb
from data.dataset import ProcessedSequenceDatasetCreator, SimpleSequenceDatasetCreator
from train_utils.evaluate import evaluate
from models.simple_cnn import SimpleCNN

from train_utils.callbacks import get_callbacks

parser = argparse.ArgumentParser()
parser.add_argument("--processed_dataset", type=bool, nargs="?", default=True)
parser.add_argument("--batch_size", type=int, nargs="?", default=4)
parser.add_argument("--time_window", type=int, nargs="?", default=8)
parser.add_argument("--input_scores", type=bool, nargs="?", default=True)
parser.add_argument("--conv2d_filters", type=int, nargs="?", default=10)
parser.add_argument("--conv1d_num", type=int, nargs="?", default=2)
parser.add_argument("--conv1d_filters", type=int, nargs="?", default=10)
parser.add_argument("--learning_rate", type=float, nargs="?", default=0.001)

args = parser.parse_args()

wandb.init(config=args)

config = wandb.config

if config["processed_dataset"]:
    dataset_creator = ProcessedSequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/",
    )
else:
    dataset_creator = SimpleSequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/",
    )

train_dataset, input_shape, class_dict = dataset_creator.get_dataset(
    "train", config["batch_size"]
)
val_dataset, *_ = dataset_creator.get_dataset("val", config["batch_size"])

model_builder = SimpleCNN(nc=len(class_dict), **config)

model = model_builder.get_model(input_shape)

callbacks, save_model_path = get_callbacks()

model.fit(
    train_dataset,
    epochs=500,
    validation_data=val_dataset,
    callbacks=callbacks,
)

evaluate(model, val_dataset, class_dict)

wandb.save(f"{save_model_path}/*")
