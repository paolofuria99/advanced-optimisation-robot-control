# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:31:22 2022

@author: Gianluigi Grandesso
"""

import math

import hw1_conf as conf
import matplotlib.pyplot as plt
import numpy as np
from plot_utils import plot_xy


def compute_3rd_order_poly_traj(x0, x1, T, dt):
    """
    Interpolates two points of the LIPM trajectory.

    For the x and z trajectory, it computes a 3rd order polynomial given the 4 following constraints:
    * it has to pass through the initial and final point;
    * it has to have zero velocity both at initial and final time.
    The velocity constraints are there in order to have smoother trajectories, especially for the z
    component.

    For the y trajectory, it is kept costant (since the robot only moves forward).

    @param x0: initial point. It can be 2-dimensional (x,y) or 1-dimensional (z)
    @param x1: final point. It can be 2-dimensional (x,y) or 1-dimensional (z)
    @param T: time step of LIPM trajectory (time the robot takes to take one full step)
    @param dt: time step of TSID trajectory
    @return: a trajectory (vector of positions, velocities, accelerations).
    If x0 and x1 are 2-dimensional, each element of the vector is 2-dimensional.
    If they are 1-dimensional, each element is 1-dimensional.
    The length of the vector is N, where N=T/dt, meaning the number of TSID time steps in one LIPM time step.
    """

    n_time_steps = int(T / dt)  # Number of TSID time steps

    if x0.shape[0] == 2:  # Computing for x,y
        x0, y0 = x0[:]
        x1, y1 = x1[:]

        x = np.zeros((2, n_time_steps))
        dx = np.zeros((2, n_time_steps))
        ddx = np.zeros((2, n_time_steps))

        # Prepare the system of 4 equations to solve
        a_matrix = np.array([[1, 0, 0, 0],
                             [1, T, T**2, T**3],
                             [0, 1, 2 * T, 3 * (T**2)],
                             [0, 1, 0, 0]])

        b_matrix = np.array([x0, x1, 0, 0])

        a, b, c, d = np.linalg.solve(a_matrix, b_matrix)

        for i in range(n_time_steps):
            t = i * dt
            x[0, i] = a + b * t + c * (t**2) + d * (t**3)  # x
            x[1, i] = y0  # y
            dx[0, i] = b + 2 * c * t + 3 * d * (t**2)  # dx
            dx[1, i] = 0.  # dy
            ddx[0, i] = 2 * c + 6 * d * t  # ddx
            ddx[1, i] = 0.  # ddy

        return x, dx, ddx
    else:  # Computing for z
        z0 = x0[0]
        z1 = x1[0]

        x = np.zeros((1, n_time_steps))
        dx = np.zeros((1, n_time_steps))
        ddx = np.zeros((1, n_time_steps))

        # Prepare the system of 4 equations to solve
        a_matrix = np.array([[1, 0, 0, 0],
                             [1, T, T**2, T**3],
                             [0, 1, 2 * T, 3 * (T**2)],
                             [0, 1, 0, 0]])

        b_matrix = np.array([z0, z1, 0, 0])

        a, b, c, d = np.linalg.solve(a_matrix, b_matrix)

        for i in range(n_time_steps):
            t = i * dt
            x[0, i] = a + b * t + c * (t**2) + d * (t**3) # z
            dx[0, i] = b + 2 * c * t + 3 * d * (t**2)  # dz
            ddx[0, i] = 2 * c + 6 * d * t  # ddz

        return x, dx, ddx


def compute_foot_traj(foot_steps, N, dt, step_time, step_height, first_phase):
    """
    @param foot_steps: array where first dimension is time step (TSID), second is (x,y)
    @param N: total number of TSID time steps
    @param dt: TSID control time interval
    @param step_time: time (in ms) to perform a robot step
    @param step_height: height of the step
    @param first_phase: stance if starting from a stance (RF), swing if coming back to a stance (LF)
    @return: a trajectory for the foot in TSID time steps (pos, vel, acc)
    """

    x = np.zeros((3, N + 1))  # Position in cartesian space (x, y, z=height)
    dx = np.zeros((3, N + 1))
    ddx = np.zeros((3, N + 1))

    N_step = int(step_time / dt)  # How many time steps in a robot step

    offset = 0

    if first_phase == 'swing':  # Left foot, initialize at first position
        x[0, :N_step] = foot_steps[0, 0]
        x[1, :N_step] = foot_steps[0, 1]
        offset = N_step

    for s in range(foot_steps.shape[0]):  # For each robot step
        i = offset + s * 2 * N_step  # 2 N_step because for each loop, you take two actions which take N_step time

        # Initialize and stay still for N_step time
        x[0, i:i + N_step] = foot_steps[s, 0]  # x pos of foot constant from i to i+N_step
        x[1, i:i + N_step] = foot_steps[s, 1]  # y pos of foot constant from i to i+N_step

        if s < foot_steps.shape[0] - 1:  # If it is not the last robot step, save next foot position in var
            next_step = foot_steps[s + 1, :]
        elif first_phase == 'swing':  # If it is the last step, and it is left foot, do nothing
            break
        else:  # If it is the last step, and it is right foot, make sure the foot is at height zero at the same x,y pos
            next_step = foot_steps[s, :]
            step_height = 0.0

        # Compute (x,y) foot traj from one position of the foot to the next
        x[:2, i + N_step: i + 2 * N_step], \
        dx[:2, i + N_step: i + 2 * N_step], \
        ddx[:2, i + N_step: i + 2 * N_step] = \
            compute_3rd_order_poly_traj(foot_steps[s, :], next_step, step_time, dt)

        # Compute z (height) foot traj from foot on the ground to max height (in half the time to take the step)
        x[2, i + N_step: i + int(1.5 * N_step)], \
        dx[2, i + N_step: i + int(1.5 * N_step)], \
        ddx[2, i + N_step: i + int(1.5 * N_step)] = \
            compute_3rd_order_poly_traj(np.array([0.]), np.array([step_height]), 0.5 * step_time, dt)

        # Compute z (height) foot traj from foot at max height to the ground (in half the time to take the step)
        x[2, i + int(1.5 * N_step):i + 2 * N_step], \
        dx[2, i + int(1.5 * N_step):i + 2 * N_step], \
        ddx[2, i + int(1.5 * N_step):i + 2 * N_step] = \
            compute_3rd_order_poly_traj(np.array([step_height]), np.array([0.0]), 0.5 * step_time, dt)

    return x, dx, ddx


def discrete_LIP_dynamics(delta_t, g, h):
    w = math.sqrt(g / h)
    A_d = np.array([[math.cosh(w * delta_t), (1 / w) * math.sinh(w * delta_t)],
                    [w * math.sinh(w * delta_t), math.cosh(w * delta_t)]])

    B_d = np.array([1 - math.cosh(w * delta_t), -w * math.sinh(w * delta_t)])

    return A_d, B_d


def interpolate_lipm_traj(T_step, nb_steps, dt_mpc, dt_ctrl, com_z, g, com_state_x, com_state_y, cop_ref, cop_x, cop_y):
    # INTERPOLATE WITH TIME STEP OF CONTROLLER (TSID)
    N = nb_steps * int(round(T_step / dt_mpc))  # number of time steps for traj-opt
    N_ctrl = int((N * dt_mpc) / dt_ctrl)  # number of time steps for TSID
    com = np.empty((3, N_ctrl + 1)) * np.nan
    dcom = np.zeros((3, N_ctrl + 1))
    ddcom = np.zeros((3, N_ctrl + 1))
    cop = np.empty((2, N_ctrl + 1)) * np.nan
    foot_steps = np.empty((2, N_ctrl + 1)) * np.nan
    contact_phase = (N_ctrl + 1) * ['right']
    com[2, :] = com_z

    N_inner = int(N_ctrl / N)
    for i in range(N):
        com[0, i * N_inner] = com_state_x[i, 0]
        com[1, i * N_inner] = com_state_y[i, 0]
        dcom[0, i * N_inner] = com_state_x[i, 1]
        dcom[1, i * N_inner] = com_state_y[i, 1]
        if i > 0:
            if np.linalg.norm(cop_ref[i, :] - cop_ref[i - 1, :]) < 1e-10:
                contact_phase[i * N_inner] = contact_phase[i * N_inner - 1]
            else:
                if contact_phase[(i - 1) * N_inner] == 'right':
                    contact_phase[i * N_inner] = 'left'
                elif contact_phase[(i - 1) * N_inner] == 'left':
                    contact_phase[i * N_inner] = 'right'

        for j in range(N_inner):
            ii = i * N_inner + j
            (A, B) = discrete_LIP_dynamics((j + 1) * dt_ctrl, g, com_z)
            foot_steps[:, ii] = cop_ref[i, :].T
            cop[0, ii] = cop_x[i]
            cop[1, ii] = cop_y[i]
            x_next = A.dot(com_state_x[i, :]) + B.dot(cop[0, ii])
            y_next = A.dot(com_state_y[i, :]) + B.dot(cop[1, ii])
            com[0, ii + 1] = x_next[0]
            com[1, ii + 1] = y_next[0]
            dcom[0, ii + 1] = x_next[1]
            dcom[1, ii + 1] = y_next[1]
            ddcom[:2, ii] = g / com_z * (com[:2, ii] - cop[:, ii])

            if j > 0:
                contact_phase[ii] = contact_phase[ii - 1]

    return com, dcom, ddcom, cop, contact_phase, foot_steps


# READ COM-COP TRAJECTORIES COMPUTED WITH LIPM MODEL
data = np.load(conf.DATA_FILE_LIPM)
com_state_x = data['com_state_x']
com_state_y = data['com_state_y']
cop_ref = data['cop_ref']
cop_x = data['cop_x']
cop_y = data['cop_y']
foot_steps = data['foot_steps']

# INTERPOLATE WITH TIME STEP OF CONTROLLER (TSID)
dt_ctrl = conf.dt  # time step used by TSID
com, dcom, ddcom, cop, contact_phase, foot_steps_ctrl = \
    interpolate_lipm_traj(conf.T_step, conf.nb_steps, conf.dt_mpc, dt_ctrl, conf.h, conf.g,
                          com_state_x, com_state_y, cop_ref, cop_x, cop_y)

# COMPUTE TRAJECTORIES FOR FEET
N = conf.nb_steps * int(round(conf.T_step / conf.dt_mpc))  # number of time steps for traj-opt
N_ctrl = int((N * conf.dt_mpc) / dt_ctrl)  # number of time steps for TSID
foot_steps_RF = foot_steps[::2, :]  # assume first footstep corresponds to right foot
x_RF, dx_RF, ddx_RF = compute_foot_traj(foot_steps_RF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'stance')
foot_steps_LF = foot_steps[1::2, :]
x_LF, dx_LF, ddx_LF = compute_foot_traj(foot_steps_LF, N_ctrl, dt_ctrl, conf.T_step, conf.step_height, 'swing')

# SAVE COMPUTED TRAJECTORIES IN NPY FILE FOR TSID
np.savez(conf.DATA_FILE_TSID, com=com, dcom=dcom, ddcom=ddcom,
         x_RF=x_RF, dx_RF=dx_RF, ddx_RF=ddx_RF,
         x_LF=x_LF, dx_LF=dx_LF, ddx_LF=ddx_LF,
         contact_phase=contact_phase, cop=cop)

# PLOT STUFF
time_ctrl = np.arange(0, round(N_ctrl * dt_ctrl, 2), dt_ctrl)

for i in range(3):
    plt.figure()
    plt.plot(time_ctrl, x_RF[i, :-1], label='x RF ' + str(i))
    plt.plot(time_ctrl, x_LF[i, :-1], label='x LF ' + str(i))
    plt.legend()

time = np.arange(0, round(N * conf.dt_mpc, 2), conf.dt_mpc)
for i in range(2):
    plt.figure()
    plt.plot(time_ctrl, cop[i, :-1], label='CoP')
    #    plt.plot(time_ctrl, foot_steps_ctrl[i,:-1], label='Foot step')
    plt.plot(time_ctrl, com[i, :-1], 'g', label='CoM')
    if i == 0:
        plt.plot(time, com_state_x[:-1, 0], ':', label='CoM TO')
    else:
        plt.plot(time, com_state_y[:-1, 0], ':', label='CoM TO')
    plt.legend()

foot_length = conf.lxn + conf.lxp  # foot size in the x-direction
foot_width = conf.lyn + conf.lyp  # foot size in the y-direciton
plot_xy(time_ctrl, N_ctrl, foot_length, foot_width,
        foot_steps_ctrl.T, cop[0, :], cop[1, :],
        com[0, :].reshape((N_ctrl + 1, 1)),
        com[1, :].reshape((N_ctrl + 1, 1)))
plt.plot(com_state_x[:, 0], com_state_y[:, 0], 'r* ', markersize=15, )
plt.gca().set_xlim([-0.2, 0.4])
plt.gca().set_ylim([-0.3, 0.3])
