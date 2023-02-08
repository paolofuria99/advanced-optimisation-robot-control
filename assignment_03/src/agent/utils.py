import numpy as np
import numpy.typing as npt


class NumpyUtils:

    @staticmethod
    def modulo_pi(array: npt.NDArray) -> npt.NDArray:
        """ Bring all elements of an array in the range [-pi, pi]. """
        return (array + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def sum_square(array: npt.NDArray) -> npt.NDArray:
        """ Compute the sum of squares of an array. """
        return np.sum(np.square(array))
