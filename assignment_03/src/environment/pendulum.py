from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.model.model as model
import tensorflow as tf
from orc.assignment_03.src.utils import NumpyUtils


class Pendulum(ABC):
    """
        A pendulum environment.
        The state space (joint angle, velocity) is continuous.
        The control space (joint torque) is discretized with the specified steps.
        Joint velocity and torque are saturated.
        Gaussian noise can be added in the dynamics.
    """

    def __init__(
            self,
            num_joints: int,
            time_step: float = 0.05,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the pendulum environment.

        Args:
            num_joints: the number of joints of the pendulum.
            time_step: the length, in seconds, of a time step.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        # Initialize a pendulum model
        self._pendulum = model.Pendulum(num_joints, max_vel, max_torque, time_step)

        # Setup attributes
        self._num_joints = num_joints
        self._time_step = time_step
        self._num_controls = num_controls
        self._max_vel = max_vel
        self._dis_res_torque = 2 * max_torque / (num_controls-1)  # Discretization resolution for joint torque
        self._rng = rng

        self._current_state: npt.NDArray = rng.random(self._num_joints * 2)

    @property
    def num_controls(self):
        return self._num_controls

    @property
    def current_state(self):
        return self._current_state

    @abstractmethod
    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        pass

    def reset(self, state: npt.NDArray = None, display: bool = False) -> npt.NDArray:
        """
        Reset the environment, either to a provided state or a random one.

        Args:
            state: the state in which you want to reset the environment. If not specified,
                a random state will be used.
            display: whether to display on Gepetto Viewer.

        Returns:
            the new state of the environment.
        """
        if state is None:
            self._current_state = self.random_state()
        else:
            self._current_state = state

        if display:
            self.render()

        return self._current_state

    def step(self, torque_idx: int, display: bool = False) -> Tuple[npt.NDArray, float]:
        """
        Perform a step by applying the given control input.

        Args:
            torque_idx: the control input to apply.
            display: whether to display on Gepetto Viewer.

        Returns:
            The new state, and the cost of applying the control input.
        """
        self._current_state, cost = self.dynamics(torque_idx)

        if display:
            self.render()

        return self.current_state, cost

    def render(self):
        """
        Display the pendulum in the current state.
        """
        self._pendulum.display(self.current_state[:self._num_joints])
        time.sleep(self._time_step)

    def dynamics(self, torque_idx: int) -> Tuple[npt.ArrayLike, float]:
        """
        Apply a discretized control input to the dynamics of the pendulum,
        to get the new state and the cost of applying the control.

        Args:
            torque_idx: the discretized control input (between 0 and control_size-1).

        Returns:
            The new state and the cost of applying the control.
        """
        torque = self.d2c_torque(torque_idx)

        self._current_state, cost = self._pendulum.dynamics(self.current_state, torque)

        return self.current_state, cost

    def render_greedy_policy(self, q_network: tf.keras.Model) -> None:
        """
        Render the greedy policy as derived by a Deep Q Network

        Args:
            q_network: the Deep Q Network model to compute the Q function.
        """
        curr_state = self.reset(display=True)

        while not self.is_goal(curr_state):
            curr_state = NumpyUtils.np_2_tf(np.array(curr_state).reshape(1, -1))
            q_values = tf.squeeze(q_network(curr_state))
            action = int(tf.argmin(q_values))
            curr_state, _ = self.step(action, display=True)

    @staticmethod
    def is_goal(state: npt.ArrayLike) -> bool:
        return (np.array(state) == 0).all()

    def random_state(self) -> npt.NDArray:
        position = self._rng.uniform(-np.pi, np.pi, self._num_joints)
        velocity = self._rng.uniform(-self._max_vel, self._max_vel, self._num_joints)
        return np.concatenate((position, velocity))


class SinglePendulum(Pendulum):

    def __init__(
            self,
            time_step: float = 0.05,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the single pendulum environment.

        Args:
            time_step: the length, in seconds, of a time step.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        super(SinglePendulum, self).__init__(
            1, time_step, num_controls, max_vel, max_torque, rng
        )

    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        torque = np.clip(torque_idx, 0, self._num_controls - 1)
        torque = torque - ((self._num_controls - 1) / 2)
        torque = torque * self._dis_res_torque
        return np.array(torque)


class DoublePendulumUnderact(Pendulum):

    def __init__(
            self,
            time_step: float = 0.05,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the underactuated double pendulum environment.

        Args:
            time_step: the length, in seconds, of a time step.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        super(DoublePendulumUnderact, self).__init__(
            2, time_step, num_controls, max_vel, max_torque, rng
        )

    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.
        The torque for the second joint is set to 0 for underactuation.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        torque = np.zeros(2)
        dis_torque = np.clip(torque_idx, 0, self._num_controls - 1)
        dis_torque = dis_torque - ((self._num_controls - 1) / 2)
        torque[0] = dis_torque * self._dis_res_torque
        return torque
