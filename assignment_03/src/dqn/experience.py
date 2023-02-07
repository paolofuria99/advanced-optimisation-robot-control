from typing import NamedTuple, Tuple

import numpy.typing as npt
import numpy as np


class Experience(NamedTuple):
    """
    This class describes an experience to be stored in the experience-replay buffer.
    """
    start_state: npt.NDArray
    action: int
    next_state: npt.NDArray
    cost: float
    goal_reached: bool


class ExperienceBuffer:
    """
    This class implements an efficient circular buffer for the experience-replay buffer.
    We make a custom class and not use something like collections.deque because we want
    both efficient circular appends (with a maximum size) and efficient random indexing,
    and the latter is not provided by deque.
    """

    def __init__(self, size: int, rng: np.random.Generator) -> None:
        self._buffer = np.empty(size, dtype=Experience)
        self._index = 0
        self._num_elements = 0
        self._size = size
        self._rng = rng

    def append(self, experience: Experience) -> None:
        """
        Append an experience to the buffer. If the buffer is full,
        the oldest experience is replaced.

        Args:
            experience: the experience to add.
        """
        self._buffer[self._index] = experience
        self._index = (self._index + 1) % self._size
        if self._num_elements < self._size:
            self._num_elements += 1

    def sample(self, quantity: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Sample uniformly a batch of experiences.

        Args:
            quantity: how many experiences to sample.

        Returns:
            A tuple containing, respectively,
            an array of all the starting states of each experience;
            an array of all the actions taken in each experience;
            an array of all the resulting states of each experience (after applying the corresponding action);
            an array of all the costs of taking an action in each experience;
            a boolean mask that says which experiences resulted in a goal state.
        """
        indices = self._rng.choice(self._num_elements, quantity, replace=False)
        start_states, actions, next_states, costs, goals = zip(*[self._buffer[idx] for idx in indices])
        return \
            np.array(start_states, dtype=np.float), \
            np.array(actions, dtype=np.int), \
            np.array(next_states, dtype=np.float), \
            np.array(costs, dtype=np.float), \
            np.array(goals, dtype=np.bool)
