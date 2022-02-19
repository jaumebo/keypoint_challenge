import numpy as np
import tensorflow as tf

import wandb

CLASS_NAMES = ["pick_up", "put_back", "raise_hand", "standing", "walking"]


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset):

    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true += tf.argmax(y, axis=-1).numpy().tolist()
        y_pred += tf.argmax(model.predict(x), axis=-1).numpy().tolist()

    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=CLASS_NAMES
            )
        }
    )

    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
