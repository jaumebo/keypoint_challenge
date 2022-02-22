import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv1D, DepthwiseConv2D
from tensorflow.keras.optimizers import Adam

from typing import Tuple


class SecondCNN(object):
    def __init__(
        self,
        nc: int = 5,
        **kwargs,
    ):
        self.nc = nc
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.input_scores = kwargs.get("input_scores", False)
        self.scale_with_scores = kwargs.get("scale_with_scores", True)
        self.position_embed_length = kwargs.get("position_embed_length", 4)
        self.time_window = kwargs.get("time_window", 10)
        self.dense_num = kwargs.get("dense_num", 1)
        self.dense_units = kwargs.get("dense_units", 10)

    def get_model(self, input_shape: Tuple[int]):

        input_tensor = tf.keras.Input(shape=input_shape)

        if not self.input_scores:
            x = input_tensor[:, :, :, :2]
            scores = input_tensor[:, :, :, 2]
        else:
            x = input_tensor

        x = tf.transpose(x, perm=[0, 2, 3, 1])

        x = DepthwiseConv2D(kernel_size=(1, x.shape[2]), activation="relu")(x)

        x = tf.squeeze(x, axis=2)

        x = tf.transpose(x, perm=[0, 2, 1])

        if not self.input_scores and self.scale_with_scores:
            x = tf.multiply(x, scores)

        x = Dense(self.position_embed_length, activation="relu")(x)

        x = Conv1D(
            self.position_embed_length,
            kernel_size=self.time_window,
            strides=self.time_window,
        )(x)

        x = Flatten()(x)

        for _ in range(self.dense_num):
            x = Dense(self.dense_units, activation="relu")(x)

        outputs = Dense(self.nc, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        return model


if __name__ == "__main__":

    simple_cnn = SecondCNN()

    simple_cnn.get_model((170, 18, 3))
