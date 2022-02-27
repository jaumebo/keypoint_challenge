import argparse

import wandb
from data.dataset import SkeletonDataset
from train_utils.evaluate import evaluate
from models.attention import SimpleAttention

from train_utils.callbacks import get_callbacks

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, nargs="?", default=0.001)
parser.add_argument("--batch_size", type=int, nargs="?", default=12)
parser.add_argument("--hidden_length", type=int, nargs="?", default=20)
parser.add_argument("--embed_length", type=int, nargs="?", default=20)
parser.add_argument("--embed_kernel", type=int, nargs="?", default=4)
parser.add_argument("--dense_num", type=int, nargs="?", default=1)
parser.add_argument("--dense_units", type=int, nargs="?", default=20)

args = parser.parse_args()

wandb.init(config=args)

config = wandb.config

dataset_creator = SkeletonDataset(
    "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/"
)

train_dataset, input_shape, class_dict = dataset_creator.get_dataset(
    "train", config["batch_size"]
)
val_dataset, *_ = dataset_creator.get_dataset("val", config["batch_size"])

model_builder = SimpleAttention(nc=len(class_dict), **config)

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
