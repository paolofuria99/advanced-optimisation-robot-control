import numpy as np
import numpy.typing as npt
import pinocchio as pin
from numpy.linalg import inv
from orc.assignment_03.src.environment.model.display import Display
from environment.utils import NumpyUtils


class Visual:
    """
    Class representing one 3D mesh of the robot, to be attached to a joint. The class contains:
        * the name of the 3D objects inside Gepetto viewer.
        * the ID of the joint in the kinematic tree to which the body is attached.
        * the placement of the body with respect to the joint frame.
    This class is only used in the list Robot.visuals (see below).
    """

    def __init__(self, name, jointParent, placement):
        self.name = name  # Name in gepetto viewer
        self.jointParent = jointParent  # ID (int) of the joint
        self.placement = placement  # placement of the body wrt joint, i.e. bodyMjoint

    def place(self, display, oMjoint):
        oMbody = oMjoint * self.placement
        display.place(self.name, oMbody, False)


class Pendulum:
    """
    Define a class Pendulum with nbJoint joints.
    The members of the class are:
        * viewer: a display encapsulating a gepetto viewer client to create 3D objects and place them.
        * model: the kinematic tree of the robot.
        * data: the temporary variables to be used by the kinematic algorithms.
        * visuals: the list of all the 'visual' 3D objects to render the robot, each element of the list being
          an object Visual (see above).
    """

    def __init__(
            self,
            num_joints: int,
            max_vel: float,
            max_torque: float,
            noise_std: float = 0.0,
            time_step: float = 5e-2,
            num_euler_steps: int = 1
    ):
        """
        Create a Pinocchio model of an N-pendulum.

        Args:
            num_joints: the number of joints of the pendulum.
            max_vel: the maximum value of the velocity.
            max_torque: the maximum value of the torque.
            noise_std: the standard deviation of the noise to be injected into the dynamics.
            time_step: the length (in seconds) of a time step.
            num_euler_steps: the number of euler steps per integration.
        """
        # Set up the model
        self._viewer = Display()
        self._visuals = []
        self._model = pin.Model()
        self._create_pendulum(num_joints)
        self._data = self._model.createData()

        # Setup parameters
        self._noise_std = noise_std
        self._time_step = time_step  # Time step length
        self._num_euler_steps = num_euler_steps  # Number of Euler steps per integration (internal)
        self._friction_coefficient = .10  # Friction coefficient
        self._max_vel = max_vel  # Max velocity (clipped if larger)
        self._max_torque = max_torque  # Max torque   (clipped if larger)

    @property
    def joint_angles_size(self):
        """ Size of the joint angles vector. """
        return self._model.nq

    """ Size of the v vector """

    @property
    def joint_velocities_size(self):
        """ Size of the joint velocities vector. """
        return self._model.nv

    @property
    def state_size(self):
        """ Size of the state vector. """
        return self.joint_angles_size + self.joint_velocities_size

    @property
    def control_size(self):
        """ Size of the control vector. """
        return self.joint_velocities_size

    def _create_pendulum(self, num_joints: int, root_id: int = 0, prefix: str = "", joint_placement=None):
        color = [1, 1, 0.78, 1.0]
        color_red = [1.0, 0.0, 0.0, 1.0]

        jointId = root_id
        jointPlacement = joint_placement if joint_placement is not None else pin.SE3.Identity()
        length = 1.0
        mass = length
        inertia = pin.Inertia(
            mass,
            np.array([0.0, 0.0, length / 2]).T,
            mass / 5 * np.diagflat([1e-2, length**2, 1e-2])
        )

        for i in range(num_joints):
            istr = str(i)
            name = prefix + "joint" + istr
            jointName = name + "_joint"
            jointId = self._model.addJoint(jointId, pin.JointModelRY(), jointPlacement, jointName)
            self._model.appendBodyToJoint(jointId, inertia, pin.SE3.Identity())
            try:
                self._viewer.viewer.gui.addSphere('world/' + prefix + 'sphere' + istr, 0.15, color_red)
            except Exception:
                pass
            self._visuals.append(Visual('world/' + prefix + 'sphere' + istr, jointId, pin.SE3.Identity()))
            try:
                self._viewer.viewer.gui.addCapsule('world/' + prefix + 'arm' + istr, .1, .8 * length, color)
            except Exception:
                pass
            self._visuals.append(
                Visual(
                    'world/' + prefix + 'arm' + istr, jointId,
                    pin.SE3(np.eye(3), np.array([0., 0., length / 2]))
                )
            )
            jointPlacement = pin.SE3(np.eye(3), np.array([0.0, 0.0, length]).T)

        self._model.addFrame(pin.Frame('tip', jointId, 0, jointPlacement, pin.FrameType.OP_FRAME))

    def display(self, joint_angles: npt.NDArray):
        """
        Display the robot in the viewer.

        Args:
            joint_angles: an array of the robot's joint angles.
        """
        pin.forwardKinematics(self._model, self._data, joint_angles)
        for visual in self._visuals:
            visual.place(self._viewer, self._data.oMi[visual.jointParent])
        self._viewer.viewer.gui.refresh()

    def dynamics(
            self,
            state: npt.NDArray,
            control: npt.NDArray
    ):
        """
        Dynamic function: state, control -> next_state.

        Args:
            state: an array of the state (1D, first N elements for joint angles,
               then N elements for joint velocities
            control: an array of the control to apply (1D, N elements, one for each joint)

        Returns:
            the next state.
        """

        q = NumpyUtils.modulo_pi(np.copy(state[:self.joint_angles_size]))
        v = np.copy(state[self.joint_angles_size:])
        u = np.clip(np.reshape(np.copy(control), self.control_size), -self._max_torque, self._max_torque)

        DT = self._time_step / self._num_euler_steps
        for i in range(self._num_euler_steps):
            pin.computeAllTerms(self._model, self._data, q, v)
            M = self._data.M
            b = self._data.nle
            a = inv(M) * (u - self._friction_coefficient * v - b)
            a = a.reshape(self.joint_velocities_size) + np.random.randn(self.joint_velocities_size) * self._noise_std

            q += (v + 0.5 * DT * a) * DT
            v += a * DT

        new_state = np.empty(self.joint_angles_size+self.joint_velocities_size)
        new_state[:self.joint_angles_size] = NumpyUtils.modulo_pi(q)
        new_state[self.joint_angles_size:] = np.clip(v, -self._max_vel, self._max_vel)

        return new_state
