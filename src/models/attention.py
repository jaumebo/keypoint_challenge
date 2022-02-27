import atexit
from lib2to3.pgen2 import token
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    Attention,
    GlobalAveragePooling1D,
    Concatenate,
    Conv1D,
    Conv3D,
)
from tensorflow.keras.optimizers import Adam


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class SimpleAttention(object):
    def __init__(
        self,
        nc: int = 5,
        **kwargs,
    ):
        self.nc = nc
        self.batch_size = kwargs.get("batch_size", 4)
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.hidden_length = kwargs.get("hidden_length", 10)
        self.embed_length = kwargs.get("embed_length", 10)
        self.embed_kernel = kwargs.get("embed_kernel", 4)
        self.dense_num = kwargs.get("dense_num", 1)
        self.dense_units = kwargs.get("dense_units", 10)

    def __get_attention_mask(self, scores):

        attention_mask = tf.math.reduce_sum(scores, axis=-1)

        attention_mask = tf.cast(attention_mask, tf.bool)

        return attention_mask

    def get_model(self, input_shape: Tuple[int]):

        input_tensor = tf.keras.Input(shape=input_shape, batch_size=self.batch_size)

        scores = input_tensor[:, :, :, -1]

        x = tf.expand_dims(input_tensor, axis=-1)

        x = Conv3D(
            filters=self.hidden_length,
            kernel_size=(1, 3, x.shape[3]),
            strides=(1, 3, 1),
            activation="relu",
        )(x)

        x = tf.squeeze(x, axis=3)

        x = tf.reshape(x, shape=(self.batch_size, x.shape[1], -1))

        x = AddPositionEmbs()(x)

        cnn_layer = Conv1D(
            filters=self.embed_length,
            kernel_size=self.embed_kernel,
            padding="same",
        )

        query = cnn_layer(x)
        key = cnn_layer(x)
        value = cnn_layer(x)
        attention_mask = self.__get_attention_mask(scores)

        query_value_attention_seq = Attention()(
            [query, value, key], mask=[attention_mask, attention_mask]
        )

        query_encoding = GlobalAveragePooling1D()(query)
        query_value_attention = GlobalAveragePooling1D()(query_value_attention_seq)
        x = Concatenate()([query_encoding, query_value_attention])

        for _ in range(self.dense_num):
            x = Dense(self.dense_units, activation="relu")(x)

        output = Dense(self.nc, activation="softmax")(x)

        model = tf.keras.Model(inputs=input_tensor, outputs=output)

        model.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        return model


if __name__ == "__main__":

    tensor = tf.random.uniform((4, 170, 12, 3))

    simple_cnn = SimpleAttention()

    model = simple_cnn.get_model((170, 12, 3))

    output = model(tensor)
    print(output.shape)

    pass
