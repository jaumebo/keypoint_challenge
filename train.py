from data.dataset import ProcessedSequenceDatasetCreator
from models.first_cnn import SimpleCNN
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import argparse
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, required=True)
parser.add_argument("--time_window", type=int, required=True)
parser.add_argument("--input_scores", type=bool, required=True)
parser.add_argument("--conv2d_filters", type=int, required=True)
parser.add_argument("--conv1d_num", type=int, required=True)
parser.add_argument("--conv1d_filters", type=int, required=True)

# Get the hyperparameters
args = parser.parse_args()

# Pass them to wandb.init
wandb.init(config=args)

# Access all hyperparameter values through wandb.config
config = wandb.config

dataset_creator = ProcessedSequenceDatasetCreator(
    "/Users/jaumebrossa/Code/AI/v7_challenge/data/trainval/"
)

train_dataset, input_shape = dataset_creator.get_dataset("train", config["batch_size"])
val_dataset, input_shape = dataset_creator.get_dataset("val", config["batch_size"])

model_builder = SimpleCNN(**config)

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

model.fit(
    train_dataset,
    epochs=1,
    validation_data=val_dataset,
    callbacks=[WandbCallback(), early_stopper, model_checkpoint],
)

evaluate(model, val_dataset)

wandb.save(f"{save_model_path}/*")
