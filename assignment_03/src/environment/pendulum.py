from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from agent.pendulum import PendulumAgent


class PendulumEnv(ABC):
    """
        A pendulum environment.
        The state space (joint angle, velocity) is continuous.
        The control space (joint torque) is discretized with the specified steps.
        Joint velocity and torque are saturated.
        Gaussian noise can be added in the dynamics.
    """

    def __init__(
            self,
            agent: PendulumAgent,
            num_controls: int = 11,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the pendulum environment.

        Args:
            num_controls: the number of discretization steps for joint torque.
            rng: a random number generator. A default one is used if not specified.
        """
        # Initialize a pendulum agent
        self._agent = agent

        # Setup attributes
        self._num_controls = num_controls
        self._rng = rng

        # Needed for converting torque from discrete to continuous
        self._dis_res_torque = 2 * agent.max_torque / (num_controls - 1)

        # Randomly initialize current state
        self._current_state = self._random_state()

    @property
    def num_controls(self):
        return self._num_controls

    @property
    def agent(self):
        return self._agent

    @property
    def current_state(self):
        return self._current_state

    @abstractmethod
    def _d2c_torque(self, torque_idx: int) -> np.typing.NDArray:
        """
        Convert a discretized torque into a continuous vector.

        Args:
            torque_idx: the torque to convert: an integer between 0 and control_size-1.

        Returns:
            The continuous torque.
        """
        pass

    def reset(self, state: npt.ArrayLike = None, display: bool = False) -> npt.NDArray:
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
            self._current_state = self._random_state()
        else:
            self._current_state = np.copy(state)

        if display:
            self.render()

        return self._current_state

    def step(self, torque_idx: int, display: bool = False) -> Tuple[npt.NDArray, float, bool]:
        """
        Perform a step by applying the given control input.

        Args:
            torque_idx: the control input to apply.
            display: whether to display on Gepetto Viewer.

        Returns:
            The new state, the cost of the step, and whether the goal was reached.
        """
        continuous_torque = self._d2c_torque(torque_idx)

        curr_state = self._current_state
        new_state, cost = self._agent.dynamics(curr_state, continuous_torque)
        goal_reached = self._is_goal(new_state)

        self._current_state = new_state

        if display:
            self.render()

        return self._current_state, cost, goal_reached

    def render(self):
        """
        Display the pendulum in the current state.
        """
        self._agent.display(self.current_state[:self._agent.num_joints])
        time.sleep(self._agent.sim_time_step)

    def render_greedy_policy(
            self,
            q_network: tf.keras.Model,
            random_start: bool = False,
            max_steps: int = None,
            num_episodes: int = 1,
            display: bool = True
    ) -> None:
        """
        Render the greedy policy as derived by a Deep Q Network.

        Args:
            q_network: the Deep Q Network agent to compute the Q function.
            random_start: whether to start from a random state. If false, it starts from bottom.
            max_steps: the maximum number of steps to take. If not given, it runs until goal is reached.
            num_episodes: the number of episodes to run.
            display: whether to render the model on the GUI.
        """
        for episode in range(num_episodes):

            if random_start:
                start_state = self._random_state()
            else:
                start_state = self._bottom_state()

            curr_state = self.reset(start_state, display=display)

            steps = 1
            while not self._is_goal(curr_state):
                curr_state_t = tf.convert_to_tensor(curr_state)
                curr_state_t = tf.expand_dims(curr_state_t, axis=0)
                q_values = tf.squeeze(q_network(curr_state_t, training=False))
                action = int(tf.argmin(q_values))
                curr_state, _, _ = self.step(action, display=display)
                if max_steps is not None and steps >= max_steps:
                    break
                steps += 1

    def _random_state(self) -> npt.NDArray:
        position = self._rng.uniform(-np.pi, np.pi, self._agent.num_joints)
        velocity = self._rng.uniform(-self._agent.max_vel, self._agent.max_vel, self._agent.num_joints)
        return np.concatenate((position, velocity))

    @staticmethod
    @abstractmethod
    def _bottom_state() -> npt.NDArray:
        pass

    @staticmethod
    def _is_goal(state: npt.NDArray) -> bool:
        return (state == 0.0).all()
