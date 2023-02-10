import numpy as np
import numpy.typing as npt
from agent.double.double_pendulum import UnderactDoublePendulumAgent
from environment.pendulum import PendulumEnv


class UnderactDoublePendulumEnv(PendulumEnv):

    def __init__(
            self,
            num_controls,
            max_vel,
            max_torque,
            rng: np.random.Generator = np.random.default_rng()
    ):
        """
        Initialize the single pendulum environment.

        Args:
            num_controls: the number of discretization steps for joint torque.
            max_vel = maximum value for joint velocity (vel in [-max_vel, max_vel]).
            max_torque: maximum value for joint torque (torque in [-max_torque, max_torque]).
            rng: a random number generator. A default one is used if not specified.
        """
        agent = UnderactDoublePendulumAgent(max_vel, max_torque)
        super(UnderactDoublePendulumEnv, self).__init__(agent, num_controls, rng)

    def d2c_torque(self, torque_idx: int) -> npt.NDArray:
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

    @staticmethod
    def _bottom_state() -> npt.NDArray:
        return np.array([-np.pi, 0.0, 0.0, 0.0])
