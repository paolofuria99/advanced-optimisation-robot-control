from dataclasses import dataclass
from typing import Tuple

import matplotlib.colors as mlp_colors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from environment.pendulum import PendulumEnv


@dataclass
class Plot:
    """
    This class takes care of plotting trajectories and cost of a pendulum simulation.
    Additionally, it can also compute and plot value and policy tables for the single
    pendulum, by discretizing the states, for visualization purposes.
    """
    states: npt.NDArray
    torques: npt.NDArray
    costs: npt.NDArray
    q_net: tf.keras.Model
    env: PendulumEnv

    def __post_init__(self):
        self._colors = ["b", "r"]
        self._time_vec = np.arange(self.states.shape[0]) * self.env.agent.sim_time_step
        self._num_discrete = 60
        # Keep only the torque of the first joint
        if self.env.agent.num_joints > 1:
            self.torques = np.array([torque[0] for torque in self.torques])
        # Pad costs and torques with a nan for the initial state
        self.costs = np.concatenate(([np.nan], self.costs))
        self.torques = np.concatenate(([np.nan], self.torques))

    def plot_all(self) -> None:
        """
        Plot the trajectories, the cost, and, for the single pendulum,
        the value and policy tables.
        """
        self._plot_trajectories()
        self._plot_costs()
        if self.env.agent.num_joints == 1:
            v_table, p_table = self._compute_value_and_policy_tables()
            self._plot_value_table(v_table)
            self._plot_policy_table(p_table)
        plt.show()

    def _plot_trajectories(self) -> None:
        """
        Plot the trajectories of the joint angles, joint velocities and torques.
        """
        angle_data = (
            np.array([state[:self.env.agent.num_joints] for state in self.states]),
            "angle",
            "[rad]"
        )
        vel_data = (
            np.array([state[self.env.agent.num_joints:] for state in self.states]),
            "velocity",
            "[rad/s]"
        )

        # Joint angles and velocities
        for data, name, y_label in [angle_data, vel_data]:
            legends = []
            plt.figure()
            for joint_idx in range(self.env.agent.num_joints):
                plt.plot(self._time_vec, data[:, joint_idx], self._colors[joint_idx])
                legends.append(f"Joint {name} {joint_idx + 1}")
            plt.gca().set_xlabel("Time [s]")
            plt.gca().set_ylabel(y_label)
            plt.legend(
                legends,
                loc='upper right'
            )

        # Torques
        plt.figure()
        plt.plot(self._time_vec, self.torques, self._colors[0])
        plt.gca().set_xlabel("Time [s]")
        plt.gca().set_ylabel("[Nm]")
        plt.legend(
            ["Joint torque 1"],
            loc='upper right'
        )

    def _plot_costs(self) -> None:
        """
        Plot the cost over the episode.
        """
        plt.figure()
        plt.plot(self._time_vec, self.costs, self._colors[0])
        plt.gca().set_xlabel("Time [s]")
        plt.gca().set_ylabel("Cost")

    def _plot_value_table(self, v_table: npt.NDArray) -> None:
        """
        Plot the given value table.
        """
        plt.figure()
        angles, vels = np.meshgrid(
            [self._d2c_angle(i) for i in range(self._num_discrete)],
            [self._d2c_vel(i) for i in range(self._num_discrete)]
        )
        plt.pcolormesh(
            angles, vels, v_table.reshape((self._num_discrete, self._num_discrete)), cmap=plt.cm.get_cmap("Blues")
        )
        plt.colorbar(label="Cost to go (value)")
        plt.title("Value table")
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")

    def _plot_policy_table(self, p_table: npt.NDArray) -> None:
        """
        Plot the given policy table.
        """
        plt.figure()
        angles, vels = np.meshgrid(
            [self._d2c_angle(i) for i in range(self._num_discrete)],
            [self._d2c_vel(i) for i in range(self._num_discrete)]
        )
        # Make a discrete color map
        cmap = plt.cm.get_cmap("RdBu")
        bounds = [self.env.d2c_torque(dis_torque) for dis_torque in range(self.env.num_controls)]
        norm = mlp_colors.BoundaryNorm(bounds, cmap.N)
        plt.pcolormesh(
            angles, vels, p_table.reshape((self._num_discrete, self._num_discrete)),
            cmap=cmap, norm=norm
        )
        plt.colorbar(label="Torque [Nm]")
        plt.title("Policy table")
        plt.xlabel("Joint angle [rad]")
        plt.ylabel("Joint velocity [rad/s]")
        plt.show()

    def _compute_value_and_policy_tables(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """
        Compute value and policy tables by discretizing the state space and using a Q-Network to compute a Q-table.
        The policy table contains the actual torques applied, instead of a number indicating the action taken.
        Returns:
            The value table and the policy table.
        """
        value_table = np.zeros(self._num_discrete**2, np.float)
        policy_table = np.zeros(self._num_discrete**2, np.float)

        states_idx = []  # Indexes of all possible discrete states
        states = []  # Continuous states computed from discrete states
        for disc_angle in range(self._num_discrete):
            for disc_vel in range(self._num_discrete):
                pair = disc_angle, disc_vel
                states_idx.append(self._2d_disc_state_to_1d_disc_state(pair))
                states.append(self._2d_disc_state_to_cont_state(pair))

        # Compute the Q-values for each possible state
        states_tf = tf.convert_to_tensor(states)
        q_table = self.q_net(states_tf, training=False)

        # Iterate on each state with its corresponding Q-values
        for state_idx, q_values in zip(states_idx, q_table):
            # Fill value table
            value_table[state_idx] = np.min(q_values)
            # For policy table, rather than simply using argmin, we do something slightly more complex
            # to ensure symmetry of the policy when multiple control inputs result in the same value.
            # In these cases we prefer the more extreme actions.
            # Also, we store the continuous torque instead of the discrete index
            u_best = np.where(q_values == value_table[state_idx])[0]
            if u_best[0] > self.env.c2d_torque(0.0):
                policy_table[state_idx] = self.env.d2c_torque(u_best[-1])
            elif u_best[-1] < self.env.c2d_torque(0.0):
                policy_table[state_idx] = self.env.d2c_torque(u_best[0])
            else:
                policy_table[state_idx] = self.env.d2c_torque(u_best[int(u_best.shape[0] / 2)])

        return value_table, policy_table

    def _2d_disc_state_to_1d_disc_state(self, disc_state: Tuple[int, int]) -> int:
        return disc_state[0] + disc_state[1] * self._num_discrete

    def _2d_disc_state_to_cont_state(self, disc_state: Tuple[int, int]) -> npt.NDArray:
        return np.array([self._d2c_angle(disc_state[0]), self._d2c_vel(disc_state[1])])

    def _d2c_angle(self, angle_idx: int) -> float:
        discr_resolution = 2 * np.pi / self._num_discrete
        angle_idx = np.clip(angle_idx, 0, self._num_discrete - 1)
        return angle_idx * discr_resolution - np.pi + 0.5 * discr_resolution

    def _d2c_vel(self, vel_idx: int) -> float:
        discr_resolution = 2 * self.env.agent.max_vel / self._num_discrete
        vel_idx = np.clip(vel_idx, 0, self._num_discrete - 1) - (self._num_discrete - 1) / 2
        return vel_idx * discr_resolution
