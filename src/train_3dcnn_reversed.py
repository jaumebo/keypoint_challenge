import argparse

import wandb
from data.dataset import ReversedBinaryAugmentedDataset, SkeletonDataset
from models.skeleton_cnn import SkeletonCNN

from train_utils.callbacks import get_callbacks
from train_utils.evaluate import evaluate

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, nargs="?", default=6)
    parser.add_argument("--time_window", type=int, nargs="?", default=10)
    parser.add_argument("--hidden_size", type=int, nargs="?", default=25)
    parser.add_argument("--num_conv2d", type=int, nargs="?", default=2)
    parser.add_argument("--num_dense", type=int, nargs="?", default=0)
    parser.add_argument("--dense_units", type=int, nargs="?", default=11)
    parser.add_argument("--learning_rate", type=float, nargs="?", default=0.001)

    args = parser.parse_args()

    wandb.init(config=args)

    config = wandb.config

    dataset_creator = ReversedBinaryAugmentedDataset(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/"
    )

    train_dataset, input_shape, class_dict = dataset_creator.get_dataset(
        "train", config["batch_size"]
    )
    val_dataset, *_ = dataset_creator.get_dataset("val", config["batch_size"])

    model_builder = SkeletonCNN(nc=len(class_dict), **config)

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
