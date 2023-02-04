from __future__ import annotations

import time
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.model.pendulum as model


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
            position: Union[npt.ArrayLike, None] = None,
            velocity: Union[npt.ArrayLike, None] = None
    ):
        self.num_joints = num_joints
        self.max_vel = max_vel
        self.position = position if position is not None else self._get_random_position()
        self.velocity = velocity if velocity is not None else self._get_random_velocity()

    @staticmethod
    def from_np(array: npt.NDArray, num_joints: int, max_vel: float) -> State:
        return State(num_joints, max_vel, position=array[:num_joints], velocity=array[num_joints:])

    def random(self):
        self.position = self._get_random_position()
        self.velocity = self._get_random_velocity()

    def is_goal(self) -> bool:
        return self.position == np.zeros(self.num_joints) and self.velocity == np.zeros(self.num_joints)

    def to_np(self) -> npt.NDArray:
        return np.concatenate((self.position, self.velocity))

    def _get_random_position(self) -> npt.ArrayLike:
        return np.random.uniform(0, 2 * np.pi, self.num_joints)

    def _get_random_velocity(self) -> npt.ArrayLike:
        return np.random.uniform(-self.max_vel, self.max_vel, self.num_joints)


class SinglePendulum:
    """
    A single pendulum environment.
    The state space (joint angle, velocity) is continuous.
    The control space (joint torque) is discretized with the specified steps.
    Joint velocity and torque are saturated.
    Gaussian noise can be added in the dynamics.
    Cost is -1 if the goal state has been reached, zero otherwise.
    """

    def __init__(
            self,
            noise_std: float = 0,
            time_step: float = 0.2,
            num_euler_steps: int = 1,
            num_controls: int = 11,
            max_vel: float = 5,
            max_torque: float = 5,
            display: bool = False
    ):
        """
        Initialize the single-pendulum environment.

        Args:
            noise_std: the standard deviation of the gaussian noise injected into the pendulum dynamics.
            time_step: the length, in seconds, of a time step.
            num_euler_steps: the number of Euler steps per integration for the pendulum dynamics.
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            display: whether to display the model on Gepetto Viewer or not.
        """
        # Initialize a pendulum model with 1 joint
        self._num_joints = 1
        self._pendulum = model.Pendulum(self._num_joints, noise_std)
        self._pendulum.DT = time_step
        self._pendulum.NDT = num_euler_steps

        # Setup attributes
        self._time_step = time_step
        self._num_controls = num_controls
        self._max_vel = max_vel
        self._max_torque = max_torque
        self._dis_res_torque = 2 * max_torque / num_controls  # Discretization resolution for joint torque
        self._state_dimensions = 1  # The dimensions of the state (joint or velocity), 1 in this case
        self._display = display

        # Randomly initialize current state
        self.current_state = State(1, self._max_vel)

    @property
    def num_controls(self):
        return self._num_controls

    def c2d_torque(self, torque: np.typing.NDArray) -> int:
        """
        Discretize a continuous torque.

        Args:
            torque: the torque array to discretize

        Returns:
            The discretized torque: an integer between 0 and control_size-1
        """
        torque = np.clip(torque, -self._max_torque + 1e-3, self._max_torque - 1e-3)
        return int(np.floor((torque + self._max_torque) / self._dis_res_torque))

    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        discrete_torque = np.clip(torque_idx, 0, self._num_controls - 1) - ((self._num_controls - 1) / 2)
        return discrete_torque * self._dis_res_torque

    def reset(self, state: State = None) -> State:
        """
        Reset the environment, either to a provided state or a random one.

        Args:
            state (optional): the state in which you want to reset the environment. If not specified,
                a random state will be used.

        Returns:
            the new state of the environment.
        """
        if state is None:
            self.current_state.random()
        else:
            self.current_state = state

        if self._display:
            self.render()

        return self.current_state

    def step(self, torque_idx: int) -> Tuple[State, float]:
        """
        Perform a step by applying the given control input.

        Args:
            torque_idx: the control input to apply.

        Returns:
            The new state, and the cost of applying the control input.
        """
        self.current_state = self.dynamics(torque_idx)
        cost = -1 if self.current_state.is_goal() else 0

        if self._display:
            self.render()

        return self.current_state, cost

    def render(self):
        """
        Display the pendulum in the current state.
        """
        self._pendulum.display(self.current_state.position)
        time.sleep(self._pendulum.DT)

    def dynamics(self, torque_idx: int) -> State:
        """
        Apply a discretized control input to the dynamics of the pendulum,
        to get the new state.

        Args:
            torque_idx: the discretized control input (between 0 and control_size-1).

        Returns:
            The new state.
        """
        torque = self.d2c_torque(torque_idx)
        new_state, _ = self._pendulum.dynamics(self.current_state.to_np(), torque)
        self.current_state = State.from_np(new_state, self._num_joints, self._max_vel)
        return self.current_state

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
