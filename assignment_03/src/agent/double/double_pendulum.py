import time
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
            max_torque: float,
            sim_time_step: float
    ):
        super(UnderactDoublePendulumAgent, self).__init__(
            2, max_vel, max_torque, sim_time_step
        )

        # Load robot agent and wrap it
        robot_data = example_robot_data.load("double_pendulum")
        self._robot = RobotWrapper(robot_data.model, robot_data.collision_model, robot_data.visual_model)

        # Simulation wrapper on the robot
        self._simu = RobotSimulator(sim_time_step, self._robot)

    @property
    def joint_angles_size(self):
        return self._robot.nq

    @property
    def joint_velocities_size(self):
        return self._robot.nv

    @property
    def state_size(self):
        return self.joint_angles_size + self.joint_velocities_size

    @property
    def control_size(self):
        return self.joint_velocities_size

    def display(self, joint_angles: npt.NDArray):
        self._simu.display(joint_angles)
        time.sleep(self._sim_time_step)

    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        dx = np.zeros(2 * self.joint_velocities_size)

        q = NumpyUtils.modulo_pi(np.copy(state[:self.joint_angles_size]))
        v = np.copy(state[self.joint_angles_size:])
        u = np.clip(np.reshape(np.copy(control), self.control_size), -self._max_torque, self._max_torque)

        ddq = pin.aba(self._robot.model, self._robot.data, q, v, u)
        dx[self.joint_velocities_size:] = ddq
        v_mean = v + 0.5 * self.sim_time_step * ddq
        dx[:self.joint_velocities_size] = v_mean

        new_state = np.copy(state) + self.sim_time_step * dx

        new_state[:self.joint_angles_size] = NumpyUtils.modulo_pi((new_state[:self.joint_angles_size]))
        new_state[self.joint_angles_size:] = np.clip(new_state[self.joint_angles_size:], -self._max_vel, self._max_vel)

        cost = self.cost_function(new_state, u)

        return new_state, cost
