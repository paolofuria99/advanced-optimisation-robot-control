from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.model.model as model
import tensorflow as tf
from orc.assignment_03.src.utils import NumpyUtils


class State:
    """
    A class that describes the state of a pendulum, in terms of:
        * Joint position
        * Joint velocity
    Both of them are arrays, which size is the number of joints of the pendulum.
    """

    def __init__(
            self,
            num_joints,
            max_vel,
            position: Union[npt.NDArray, None] = None,
            velocity: Union[npt.NDArray, None] = None,
            rng: np.random.Generator = np.random.default_rng()
    ):
        self._rng = rng
        self._num_joints = num_joints
        self._max_vel = max_vel
        self._position = position if position is not None else self._get_random_position()
        self._velocity = velocity if velocity is not None else self._get_random_velocity()

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    def set(self, position: npt.NDArray, velocity: npt.NDArray) -> None:
        self._position = position
        self._velocity = velocity

    def random(self):
        self._position = self._get_random_position()
        self._velocity = self._get_random_velocity()

    def is_goal(self) -> bool:
        return (self._position == np.zeros(self._num_joints)).all() \
               and (self._velocity == np.zeros(self._num_joints)).all()

    def to_np(self) -> npt.NDArray:
        return np.concatenate((self._position, self._velocity))

    def _get_random_position(self) -> npt.ArrayLike:
        return self._rng.uniform(-np.pi, np.pi, self._num_joints)

    def _get_random_velocity(self) -> npt.ArrayLike:
        return self._rng.uniform(-self._max_vel, self._max_vel, self._num_joints)


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
            noise_std: float = 0,
            time_step: float = 0.05,
            num_euler_steps: int = 1,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the pendulum environment.

        Args:
            num_joints: the number of joints of the pendulum.
            noise_std: the standard deviation of the gaussian noise injected into the pendulum dynamics.
            time_step: the length, in seconds, of a time step.
            num_euler_steps: the number of Euler steps per integration for the pendulum dynamics.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        # Initialize a pendulum model
        self._pendulum = model.Pendulum(num_joints, max_vel, max_torque, noise_std, time_step, num_euler_steps)

        # Setup attributes
        self._num_joints = num_joints
        self._time_step = time_step
        self._num_controls = num_controls
        self._max_vel = max_vel
        self._max_torque = max_torque
        self._dis_res_torque = 2 * max_torque / num_controls  # Discretization resolution for joint torque
        self._rng = rng

        # Randomly initialize current state
        self.current_state = State(num_joints, max_vel, rng=rng)

    @property
    def num_controls(self):
        return self._num_controls

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

    def reset(self, state: State = None, display: bool = False) -> State:
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
            self.current_state.random()
        else:
            self.current_state.set(state.position, state.velocity)

        if display:
            self.render()

        return self.current_state

    def step(self, torque_idx: int, display: bool = False) -> Tuple[State, float]:
        """
        Perform a step by applying the given control input.

        Args:
            torque_idx: the control input to apply.
            display: whether to display on Gepetto Viewer.

        Returns:
            The new state, and the cost of applying the control input.
        """
        self.current_state, cost = self.dynamics(torque_idx)

        if display:
            self.render()

        return self.current_state, cost

    def render(self):
        """
        Display the pendulum in the current state.
        """
        self._pendulum.display(self.current_state.position)
        time.sleep(self._time_step)

    def dynamics(self, torque_idx: int) -> Tuple[State, float]:
        """
        Apply a discretized control input to the dynamics of the pendulum,
        to get the new state and the cost of applying the control.

        Args:
            torque_idx: the discretized control input (between 0 and control_size-1).

        Returns:
            The new state and the cost of applying the control.
        """
        torque = self.d2c_torque(torque_idx)
        new_state, cost = self._pendulum.dynamics(self.current_state.to_np(), torque)

        self.current_state.set(new_state[:self._num_joints], new_state[self._num_joints:])

        return self.current_state, cost

    def render_greedy_policy(self, q_network: tf.keras.Model) -> None:
        """
        Render the greedy policy as derived by a Deep Q Network

        Args:
            q_network: the Deep Q Network model to compute the Q function.
        """
        curr_state = self.reset(display=True)

        while not curr_state.is_goal():
            curr_state = NumpyUtils.np_2_tf(curr_state.to_np().reshape(1, -1))
            q_values = tf.squeeze(q_network(curr_state))
            action = int(tf.argmax(q_values))
            curr_state, _ = self.step(action, display=True)

    @staticmethod
    def plot_V_table(v_table: np.typing.NDArray) -> None:
        """
        Plot the given value table.

        Args:
            v_table: the value table to plot.
        """
        pass

    @staticmethod
    def plot_policy(policy_table: np.typing.NDArray):
        """
        Plot the given policy table.

        Args:
            policy_table: the policy table to plot.
        """
        pass

    @staticmethod
    def plot_Q_table(q_table: np.typing.NDArray):
        """
        Plot the given Q table
        Args:
            q_table: the Q table to plot
        """
        pass


class SinglePendulum(Pendulum):

    def __init__(
            self,
            noise_std: float = 0,
            time_step: float = 0.05,
            num_euler_steps: int = 1,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the single pendulum environment.

        Args:
            noise_std: the standard deviation of the gaussian noise injected into the pendulum dynamics.
            time_step: the length, in seconds, of a time step.
            num_euler_steps: the number of Euler steps per integration for the pendulum dynamics.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        super(SinglePendulum, self).__init__(
            1, noise_std, time_step, num_euler_steps, num_controls, max_vel, max_torque, rng
        )

    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        discrete_torque = np.clip(torque_idx, 0, self._num_controls - 1) - ((self._num_controls - 1) / 2)
        torque = discrete_torque * self._dis_res_torque
        return np.array(torque)


class DoublePendulumUnderact(Pendulum):

    def __init__(
            self,
            noise_std: float = 0,
            time_step: float = 0.05,
            num_euler_steps: int = 1,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the underactuated double pendulum environment.

        Args:
            noise_std: the standard deviation of the gaussian noise injected into the pendulum dynamics.
            time_step: the length, in seconds, of a time step.
            num_euler_steps: the number of Euler steps per integration for the pendulum dynamics.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        super(DoublePendulumUnderact, self).__init__(
            2, noise_std, time_step, num_euler_steps, num_controls, max_vel, max_torque, rng
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
        discrete_torque = np.clip(torque_idx, 0, self._num_controls - 1) - ((self._num_controls - 1) / 2)
        torque[0] = discrete_torque * self._dis_res_torque
        return torque
