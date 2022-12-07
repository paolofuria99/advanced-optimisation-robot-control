from typing import Tuple

import numpy as np


def mean_squared_error(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    Computes mean squared error between two vectors
    """

    return ((vector1 - vector2)**2).mean()


def load_traj(prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads trajectories (states and controls) from .npy files
    @param prefix: prefix of the file (sel, pen)
    @return: state trajectory (X) and control trajectory (U)
    """

    X = np.load(prefix+"_X.npy")
    U = np.load(prefix+"_U.npy")
    return X, U


def main():
    """
    Method used to compute the difference between the trajectories found by DPP when using the
    selection matrix (sel) and penalty (pen) methods
    """

    X_sel, U_sel = load_traj("sel")
    X_pen, U_pen = load_traj("pen")
    print("\n" + " MEAN SQUARED ERROR BETWEEN SELECTION AND PENALTY REFERENCE TRAJECTORIES ".center(60, '*'))
    print("Joint 1 pos = ", mean_squared_error(X_sel[:, 0], X_pen[:, 0]))
    print("Joint 2 pos = ", mean_squared_error(X_sel[:, 1], X_pen[:, 1]))
    print("Joint 1 vel = ", mean_squared_error(U_sel[:, 0], U_pen[:, 0]))
    print("Joint 2 vel = ", mean_squared_error(X_sel[:, 2], X_pen[:, 2]))
    print("Joint 1 torque = ", mean_squared_error(X_sel[:, 3], X_pen[:, 3]))


if __name__ == "__main__":
    main()
