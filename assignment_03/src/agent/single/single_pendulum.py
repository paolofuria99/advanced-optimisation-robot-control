from typing import Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from agent.pendulum import PendulumAgent
from agent.single.display import Display, Visual
from agent.utils import NumpyUtils
from numpy.linalg import inv


class SinglePendulumAgent(PendulumAgent):

    def __init__(
            self,
            max_vel: float,
            max_torque: float,
            sim_time_step: float
    ):

        super(SinglePendulumAgent, self).__init__(
            1, max_vel, max_torque, sim_time_step
        )

        # Set up the agent
        self._viewer = Display()
        self._visuals = []
        self._model = pin.Model()

        self._create_pendulum(1)
        self._data = self._model.createData()

        # Setup friction
        self._friction_coefficient = .10

    @property
    def joint_angles_size(self):
        return self._model.nq

    @property
    def joint_velocities_size(self):
        return self._model.nv

    @property
    def state_size(self):
        return self.joint_angles_size + self.joint_velocities_size

    @property
    def control_size(self):
        return self.joint_velocities_size

    def _create_pendulum(self, num_joints: int):
        color = [1, 1, 0.78, 1.0]
        color_red = [1.0, 0.0, 0.0, 1.0]

        jointId = 0
        jointPlacement = pin.SE3.Identity()
        length = 1.0
        mass = length
        inertia = pin.Inertia(
            mass,
            np.array([0.0, 0.0, length / 2]).T,
            mass / 5 * np.diagflat([1e-2, length**2, 1e-2])
        )

        for i in range(num_joints):
            istr = str(i)
            name = "joint" + istr
            jointName = name + "_joint"
            jointId = self._model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
            self._model.appendBodyToJoint(jointId, inertia, pin.SE3.Identity())
            try:
                self._viewer.viewer.gui.addSphere('world/' + 'sphere' + istr, 0.15, color_red)
            except Exception:
                pass
            self._visuals.append(Visual('world/' + 'sphere' + istr, jointId, pin.SE3.Identity()))
            try:
                self._viewer.viewer.gui.addCapsule('world/' + 'arm' + istr, .1, .8 * length, color)
            except Exception:
                pass
            self._visuals.append(
                Visual(
                    'world/' + 'arm' + istr, jointId,
                    pin.SE3(np.eye(3), np.array([0., 0., length / 2]))
                )
            )
            jointPlacement = pin.SE3(np.eye(3), np.array([0.0, 0.0, length]).T)

        self._model.addFrame(pin.Frame('tip', jointId, 0, jointPlacement, pin.FrameType.OP_FRAME))

    def display(self, joint_angles: npt.NDArray) -> None:
        pin.forwardKinematics(self._model, self._data, joint_angles)
        for visual in self._visuals:
            visual.place(self._viewer, self._data.oMi[visual.jointParent])
        self._viewer.viewer.gui.refresh()

    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ) -> Tuple[npt.NDArray, float]:
        q = NumpyUtils.modulo_pi(np.copy(state[:self.joint_angles_size]))
        v = np.clip(np.copy(state[self.joint_angles_size:]), -self._max_vel, self._max_vel)
        u = np.clip(np.copy(control), -self._max_torque, self._max_torque)

        pin.computeAllTerms(self._model, self._data, q, v)
        M = self._data.M
        b = self._data.nle
        a = inv(M) * (u - self._friction_coefficient * v - b)
        a = a.reshape(self.joint_velocities_size)

        DT = self._sim_time_step
        q += (v + 0.5 * DT * a) * DT
        v += a * DT

        new_state = np.empty(self.state_size)
        new_state[:self.joint_angles_size] = NumpyUtils.modulo_pi(q)
        new_state[self.joint_angles_size:] = np.clip(v, -self._max_vel, self._max_vel)

        cost = self.cost_function(new_state, u)

        return new_state, cost
