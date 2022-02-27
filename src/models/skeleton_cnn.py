import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Conv3D
from tensorflow.keras.optimizers import Adam

from typing import Tuple


class SkeletonCNN(object):
    def __init__(
        self,
        nc: int = 5,
        **kwargs,
    ):
        self.nc = nc
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.time_window = kwargs.get("time_window", 4)
        self.hidden_size = kwargs.get("hidden_size", 5)
        self.num_conv2d = kwargs.get("num_conv2d", 2)
        self.num_dense = kwargs.get("num_dense", 2)
        self.dense_units = kwargs.get("dense_units", 20)

    def get_model(self, input_shape: Tuple[int]):

        input_tensor = tf.keras.Input(shape=input_shape)

        x = tf.expand_dims(input_tensor, axis=-1)

        x = Conv3D(
            filters=self.hidden_size,
            kernel_size=(self.time_window, 3, x.shape[3]),
            strides=(self.time_window, 3, 1),
            activation="relu",
        )(x)

        x = tf.squeeze(x, axis=3)

        for _ in range(self.num_conv2d):
            x = Conv2D(
                self.hidden_size, kernel_size=(4, 4), padding="same", activation="relu"
            )(x)

        x = Flatten()(x)

        for _ in range(self.num_dense):
            x = Dense(self.dense_units, activation="relu")(x)

        output = Dense(self.nc, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        return model
