import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv1D

from typing import Tuple


class SimpleCNN(object):
    def __init__(
        self,
        nc: int = 5,
        **kwargs,
    ):
        self.nc = nc
        self.time_window = kwargs.get("time_window", 3)
        self.input_scores = kwargs.get("input_scores", False)
        self.conv2d_filters = kwargs.get("conv2d_filters", 2)
        self.conv1d_num = kwargs.get("conv1d_num", 2)
        self.conv1d_filters = kwargs.get("conv1d_filters", 2)

    def get_model(self, input_shape: Tuple[int]):

        input_tensor = tf.keras.Input(shape=input_shape)

        if not self.input_scores:
            x = input_tensor[:, :, :, :2]
        else:
            x = input_tensor

        x = Conv2D(
            self.conv2d_filters,
            kernel_size=(self.time_window, input_shape[1]),
            activation="relu",
        )(x)
        x = tf.squeeze(x, axis=2)

        for _ in range(self.conv1d_num):
            x = Conv1D(
                self.conv1d_filters,
                kernel_size=self.time_window,
                padding="same",
                activation="relu",
            )(x)

        x = Flatten()(x)

        x = Dense(self.nc, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=x)

        model.compile(
            optimizer="Adam", loss="categorical_crossentropy", metrics=["acc"]
        )

        return model


if __name__ == "__main__":

    simple_cnn = SimpleCNN()

    simple_cnn.get_model((169, 18, 3))
