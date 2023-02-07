import tensorflow as tf


class Network:
    """
    This is a static utility class used to get the network model to be used
    in the Deep Q Learning algorithm.
    """

    @staticmethod
    def get_model(input_size: int, output_size: int) -> tf.keras.Model:
        """
        Get the neural network model to use for Deep Q Learning.

        Args:
            input_size: the number of neurons in the input layer
            output_size: the number of neurons in the output layer

        Returns:
            A neural network model.
        """
        inputs = tf.keras.layers.Input(input_size)
        state_out1 = tf.keras.layers.Dense(24, activation="relu", kernel_initializer="variance_scaling")(inputs)
        state_out2 = tf.keras.layers.Dense(48, activation="relu", kernel_initializer="variance_scaling")(state_out1)
        state_out3 = tf.keras.layers.Dense(48, activation="relu", kernel_initializer="variance_scaling")(state_out2)
        state_out4 = tf.keras.layers.Dense(24, activation="relu", kernel_initializer="variance_scaling")(state_out3)
        outputs = tf.keras.layers.Dense(output_size)(state_out4)

        return tf.keras.Model(inputs=inputs, outputs=outputs)
