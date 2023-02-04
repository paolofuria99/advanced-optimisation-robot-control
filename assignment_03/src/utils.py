import tensorflow as tf
import numpy.typing as npt


class Utils:

    @staticmethod
    def np_2_tf(y) -> tf.Tensor:
        """ Convert from numpy to tensorflow """
        return tf.convert_to_tensor(y)

    @staticmethod
    def tf_2_np(y) -> npt.NDArray:
        """ Convert from tensorflow to numpy """
        return y.numpy()
