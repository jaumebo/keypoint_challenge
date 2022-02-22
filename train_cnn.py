from data.dataset import ProcessedSequenceDatasetCreator, SimpleSequenceDatasetCreator
from models.first_cnn import SimpleCNN
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import argparse
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--processed_dataset", type=bool, nargs="?", default=True)
parser.add_argument("--batch_size", type=int, nargs="?", default=4)
parser.add_argument("--time_window", type=int, nargs="?", default=8)
parser.add_argument("--input_scores", type=bool, nargs="?", default=True)
parser.add_argument("--conv2d_filters", type=int, nargs="?", default=10)
parser.add_argument("--conv1d_num", type=int, nargs="?", default=2)
parser.add_argument("--conv1d_filters", type=int, nargs="?", default=10)
parser.add_argument("--learning_rate", type=float, nargs="?", default=0.001)

# Get the hyperparameters
args = parser.parse_args()

# Pass them to wandb.init
wandb.init(config=args)

# Access all hyperparameter values through wandb.config
config = wandb.config

if config["processed_dataset"]:
    dataset_creator = ProcessedSequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/",
        classes=["pick_up", "put_back"],
    )
else:
    dataset_creator = SimpleSequenceDatasetCreator(
        "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/",
        classes=["pick_up", "put_back"],
    )

train_dataset, input_shape, class_dict = dataset_creator.get_dataset(
    "train", config["batch_size"]
)
val_dataset, *_ = dataset_creator.get_dataset("val", config["batch_size"])

model_builder = SimpleCNN(nc=len(class_dict), **config)

model = model_builder.get_model(input_shape)

early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, restore_best_weights=True
)

save_model_path = f"runs/{wandb.run.name}"

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    save_model_path,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
)

learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=10,
)

model.fit(
    train_dataset,
    epochs=500,
    validation_data=val_dataset,
    callbacks=[
        WandbCallback(),
        early_stopper,
        model_checkpoint,
        learning_rate_scheduler,
    ],
)

evaluate(model, val_dataset, class_dict)

wandb.save(f"{save_model_path}/*")
