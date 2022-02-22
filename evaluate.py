import numpy as np
import tensorflow as tf

import wandb


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, class_dict: dict):

    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true += tf.argmax(y, axis=-1).numpy().tolist()
        y_pred += tf.argmax(model.predict(x), axis=-1).numpy().tolist()

    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=list(class_dict.keys())
            )
        }
    )

    wandb.sklearn.plot_confusion_matrix(y_true, y_pred, list(class_dict.keys()))
