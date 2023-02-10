import numpy as np
import numpy.typing as npt
from agent.single.single_pendulum import SinglePendulumAgent
from environment.pendulum import PendulumEnv


class SinglePendulumEnv(PendulumEnv):

    def __init__(
            self,
            num_controls,
            max_vel,
            max_torque,
            rng: np.random.Generator = np.random.default_rng()
    ) -> None:
        """
        Initialize the single pendulum environment.

        Args:
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        agent = SinglePendulumAgent(max_vel, max_torque)
        super(SinglePendulumEnv, self).__init__(agent, num_controls, rng)

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

    @staticmethod
    def _bottom_state() -> npt.NDArray:
        return np.array([-np.pi, 0.0])
