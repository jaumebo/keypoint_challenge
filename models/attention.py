import atexit
from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import (
    Concatenate,
    Dense,
    DepthwiseConv2D,
    GlobalAveragePooling1D,
    MultiHeadAttention,
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
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.num_heads = kwargs.get("num_heads", 4)
        self.key_dim = kwargs.get("key_dim", 10)
        self.dense_num = kwargs.get("dense_num", 1)
        self.dense_units = kwargs.get("dense_units", 10)

    def __get_attention_mask(self, scores: tf.Tensor):

        scores_list = tf.reduce_sum(scores, axis=1)

        attention_mask = tf.stack(
            [scores_list for _ in range(scores.shape[-1])], axis=1
        )

        attention_mask *= tf.transpose(attention_mask, perm=[0, 2, 1])

        attention_mask = tf.cast(attention_mask, tf.bool)

        return attention_mask

    def get_model(self, input_shape: Tuple[int]):

        input_tensor = tf.keras.Input(shape=input_shape)

        x = input_tensor[:, :, :, :2]
        scores = input_tensor[:, :, :, 2]

        x = tf.transpose(x, perm=[0, 2, 3, 1])
        scores = tf.transpose(scores, perm=[0, 2, 1])

        x = DepthwiseConv2D(kernel_size=(1, x.shape[2]), activation="relu")(x)

        x = tf.squeeze(x, axis=2)

        x *= scores

        x = tf.transpose(x, perm=[0, 2, 1])

        x = AddPositionEmbs()(x)

        # query = Dense(input_shape[0], activation="linear")(x)
        # key = Dense(input_shape[0], activation="linear")(x)
        # value = Dense(input_shape[0], activation="linear")(x)
        attention_mask = self.__get_attention_mask(scores)

        query_value_attention_seq = MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )(query=x, value=x, key=x, attention_mask=attention_mask)

        query_encoding = GlobalAveragePooling1D()(x)
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

    tensor = tf.random.uniform((4, 170, 13, 3))

    simple_cnn = SimpleAttention()

    model = simple_cnn.get_model((170, 13, 3))

    output = model(tensor)
    print(output.shape)

    pass
