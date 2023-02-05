import tensorflow as tf
import numpy.typing as npt
import numpy as np


class NumpyUtils:

    @staticmethod
    def np_2_tf(array: npt.NDArray) -> tf.Tensor:
        """ Convert from numpy to tensorflow. """
        return tf.convert_to_tensor(array)

    @staticmethod
    def tf_2_np(tensor: tf.Tensor) -> npt.NDArray:
        """ Convert from tensorflow to numpy. """
        return tensor.numpy()

    @staticmethod
    def modulo_pi(array: npt.NDArray) -> npt.NDArray:
        """ Bring all elements of an array in the range [-pi, pi]. """
        return (array + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def sum_square(array: npt.NDArray) -> npt.NDArray:
        """ Compute the sum of squares of an array. """
        return np.sum(np.square(array))
