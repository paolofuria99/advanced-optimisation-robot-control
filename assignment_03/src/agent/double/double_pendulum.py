from typing import Tuple

import example_robot_data
import numpy as np
import numpy.typing as npt
import pinocchio as pin
from agent.double.simulator import RobotSimulator
from agent.double.wrapper import RobotWrapper
from agent.pendulum import PendulumAgent
from agent.utils import NumpyUtils


class UnderactDoublePendulumAgent(PendulumAgent):

    def __init__(
            self,
            max_vel: float,
            max_torque: float
    ):
        super(UnderactDoublePendulumAgent, self).__init__(
            2, max_vel, max_torque, 5e-3
        )

        # Load robot agent and wrap it
        robot_data = example_robot_data.load("double_pendulum")
        self._robot = RobotWrapper(robot_data.model, robot_data.collision_model, robot_data.visual_model)

        # Simulation wrapper on the robot
        self._simu = RobotSimulator(self._sim_time_step, self._robot)

    def display(self, joint_angles: npt.NDArray):
        self._simu.display(joint_angles)

    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        dx = np.zeros(2 * self._num_joints)

        q = NumpyUtils.modulo_pi(np.copy(state[:self._num_joints]))
        v = np.copy(state[self._num_joints:])
        u = np.clip(np.reshape(np.copy(control), self._num_joints), -self._max_torque, self._max_torque)

        ddq = pin.aba(self._robot.model, self._robot.data, q, v, u)
        dx[self._num_joints:] = ddq
        v_mean = v + 0.5 * self.sim_time_step * ddq
        dx[:self._num_joints] = v_mean

        new_state = np.copy(state) + self.sim_time_step * dx

        new_state[:self._num_joints] = NumpyUtils.modulo_pi((new_state[:self._num_joints]))
        new_state[self._num_joints:] = np.clip(new_state[self._num_joints:], -self._max_vel, self._max_vel)

        cost = self.cost_function(new_state, u)

        return new_state, cost

    def cost_function(self, state: npt.NDArray, torque: npt.NDArray) -> float:
        angle = state[:self._num_joints]
        velocity = state[self._num_joints:]

        return \
            + NumpyUtils.sum_square(angle[0] + angle[1]) \
            + 0.85 * NumpyUtils.sum_square(angle[0]) \
            + 1e-1 * NumpyUtils.sum_square(angle[1]) \
            + 1e-2 * NumpyUtils.sum_square(velocity[0])
