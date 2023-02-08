from abc import ABC, abstractmethod
from typing import Tuple

import numpy.typing as npt
from agent.utils import NumpyUtils


class PendulumAgent(ABC):

    def __init__(
            self,
            num_joints: int,
            max_vel: float,
            max_torque: float,
            time_step: float
    ) -> None:
        self._num_joints = num_joints
        self._max_vel = max_vel
        self._max_torque = max_torque
        self._sim_time_step = time_step

    @property
    def num_joints(self):
        return self._num_joints

    @property
    def max_vel(self):
        return self._max_vel

    @property
    def max_torque(self):
        return self._max_torque

    @property
    def sim_time_step(self):
        return self._sim_time_step

    @abstractmethod
    def display(self, joint_angles: npt.NDArray) -> None:
        """
        Display the robot in the viewer.

        Args:
            joint_angles: an array of the robot's joint angles.
        """
        pass

    @abstractmethod
    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        """
        Dynamic function: state, control -> next_state.

        Args:
            state: an array of the state (1D, first N elements for joint angles,
               then N elements for joint velocities
            control: an array of the control to apply (1D, N elements, one for each joint)

        Returns:
            the next state and the cost of taking the action.
        """
        pass

    def cost_function(self, state: npt.NDArray, torque: npt.NDArray) -> float:
        angle = state[:self._num_joints]
        velocity = state[self._num_joints:]

        return \
            NumpyUtils.sum_square(angle) \
            + 1e-1 * NumpyUtils.sum_square(velocity) \
            + 1e-3 * NumpyUtils.sum_square(torque)
