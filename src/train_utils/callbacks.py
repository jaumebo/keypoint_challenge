import tensorflow as tf
import wandb
from wandb.keras import WandbCallback


def get_callbacks():

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

    callbacks = [
        WandbCallback(),
        early_stopper,
        model_checkpoint,
        learning_rate_scheduler,
    ]

    return callbacks, save_model_path
