from __future__ import annotations

import time
from typing import Union, Tuple

import numpy as np
import numpy.typing as npt
from orc.assignment_03.src.environment.model.pendulum import Pendulum


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

    def random(self):
        self.position = self._get_random_position()
        self.velocity = self._get_random_velocity()

    def _get_random_position(self) -> npt.ArrayLike:
        return np.random.uniform(0, 2*np.pi, self.num_joints)

    def _get_random_velocity(self) -> npt.ArrayLike:
        return np.random.uniform(-self.max_vel, self.max_vel, self.num_joints)

    def is_goal(self) -> bool:
        return self.position == np.zeros(self.num_joints) and self.velocity == np.zeros(self.num_joints)


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
            number_euler_steps: int = 1,
            control_size: int = 11,
            max_vel: float = 5,
            max_torque: float = 5
    ):
        """
        Initialize the single-pendulum environment.

        Args:
            noise_std: the standard deviation of the gaussian noise injected into the pendulum dynamics.
            time_step: the length, in seconds, of a time step.
            number_euler_steps: the number of Euler steps per integration for the pendulum dynamics.
            control_size: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
        """
        # Initialize a pendulum model with 1 joint
        self.pendulum = Pendulum(1, noise_std)
        self.pendulum.DT = time_step
        self.pendulum.NDT = number_euler_steps

        # Setup attributes
        self.time_step = time_step
        self.control_size = control_size
        self.max_vel = max_vel
        self.max_torque = max_torque
        self._dis_res_torque = 2*max_torque/control_size  # Discretization resolution for joint torque
        self._state_dimensions = 1  # The dimensions of the state (joint or velocity), 1 in this case

        # Randomly initialize current state
        self.current_state = State(1, self.max_vel)

    def c2d_torque(self, torque: np.typing.NDArray) -> int:
        """
        Discretize a continuous torque.

        Args:
            torque: the torque array to discretize

        Returns:
            The discretized torque: an integer between 0 and control_size-1
        """
        torque = np.clip(torque, -self.max_torque + 1e-3, self.max_torque - 1e-3)
        return int(np.floor((torque + self.max_torque)/self._dis_res_torque))

    def d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        discrete_torque = np.clip(torque_idx, 0, self.control_size - 1) - ((self.control_size - 1)/2)
        return discrete_torque*self._dis_res_torque

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
        return self.current_state, cost

    def render(self):
        """
        Display the pendulum in the current state.
        """
        self.pendulum.display(self.current_state.position)
        time.sleep(self.pendulum.DT)

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
        self.current_state, _ = self.pendulum.dynamics(self.current_state, torque)
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
