from __future__ import annotations

import collections
import json
import os
import time
from datetime import datetime
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from dqn.experience import ExperienceBuffer, Experience
from dqn.network import Network
from environment.double_pendulum import UnderactDoublePendulumEnv
from environment.pendulum import PendulumEnv
from environment.single_pendulum import SinglePendulumEnv
from tensorflow.python.ops.numpy_ops import np_config


class DQL:
    # Folder to save data
    save_folder = "models"

    class HyperParams(NamedTuple):
        """
        This class describes the various hyperparameters needed in a Deep-Q-Network setting.
        """
        replay_size: int
        replay_start: int
        discount: float
        max_episodes: int
        max_steps_per_episode: int
        steps_for_target_update: int
        epsilon_start: float
        epsilon_decay: float
        epsilon_min: float
        batch_size: int
        learning_rate: float
        display_every_episodes: int

    def __init__(
            self,
            name: str,
            network_model: tf.keras.Model,
            hyper_params: HyperParams,
            env: PendulumEnv,
            rng: np.random.Generator = np.random.default_rng()
    ) -> None:
        self._name = name
        self._q_network = tf.keras.models.clone_model(network_model)
        self._q_target = tf.keras.models.clone_model(network_model)
        self._hyper_params = hyper_params
        self._env = env

        # Set the rng
        self._rng = rng

        # Prepare the experience replay buffer
        self._exp_buffer = ExperienceBuffer(self._hyper_params.replay_size, self._rng)

        # Folder for saving and logging purposes
        self._model_folder = f"{self.save_folder}/{name}"
        if os.path.exists(self._model_folder):
            # Create a sub-folder to avoid deleting precious files by mistake
            curr_time = datetime.now().strftime("%H%M%S")
            self._model_folder += f"/{curr_time}"
        os.mkdir(self._model_folder)

        # Log parameters
        self.save_params()

    @property
    def model_folder(self) -> str:
        return self._model_folder

    def train(self) -> tf.keras.Model:
        """
        Train the Q-network.

        Returns:
            The Q-network from the last episode. The best agent is saved separately..
        """
        np_config.enable_numpy_behavior()

        # Set epsilon to its starting value, which will be decayed
        epsilon = self._hyper_params.epsilon_start

        # Keep track of useful data
        total_steps = 0
        episodes_time = []
        episodes_costs = []
        episodes_losses = []

        # To decide if to save a model, we compute the average cost to go over a number
        # of last episodes. We save the model when the cost to go improves
        avg_cost_to_go_window = 5
        last_costs_to_go = collections.deque(maxlen=avg_cost_to_go_window)
        best_avg_cost_to_go = np.inf

        # Optimizer for the training
        optimizer = tf.keras.optimizers.Adam(self._hyper_params.learning_rate)

        # Run training for a maximum number of episodes
        for episode in range(1, self._hyper_params.max_episodes + 1):

            print("======================")
            print(f"EPISODE {episode}")
            print("======================")

            # Flag to know if to display or not
            display = (episode % self._hyper_params.display_every_episodes) == 0

            # Initialize variables to keep track of progress
            goal_reached = False
            episode_costs = []
            episode_losses = []

            # Set the environment to a random state
            self._env.reset(display=display)

            # Start calculating time for each episode
            start_time = time.time()

            # Run each episode for a maximum number of steps (or until the state is terminal)
            for step in range(self._hyper_params.max_steps_per_episode):

                total_steps += 1

                # Decay the epsilon
                epsilon = max(epsilon * self._hyper_params.epsilon_decay, self._hyper_params.epsilon_min)

                # Choose an action and perform a step from the current state
                curr_state = self._env.current_state
                action = self.choose_action(curr_state, epsilon)
                new_state, cost, goal_reached = self._env.step(action, display=display)

                episode_costs.append(cost)

                # Store the experience in the experience-replay buffer
                self._exp_buffer.append(
                    Experience(curr_state, action, new_state, cost, goal_reached)
                )

                # Perform a training step if enough steps have been performed
                loss = 0.0
                if total_steps >= self._hyper_params.replay_start:
                    loss = self.training_step(optimizer)
                episode_losses.append(loss)

                # Copy weights to target network every certain number of steps
                if total_steps % self._hyper_params.steps_for_target_update == 0:
                    print(f"\t Copying weights to target network.")
                    self._q_target.set_weights(self._q_network.get_weights())

                # End the episode if reached the goal
                if goal_reached:
                    break

            end_time = time.time()

            # Compute elapsed time and keep track of it
            episode_time = end_time - start_time
            episodes_time.append(episode_time)

            # Keep track of episode costs and losses
            episodes_costs.append(episode_costs)
            episodes_losses.append(episode_losses)

            # Compute cost to go
            discounted_episode_costs = [
                cost * (self._hyper_params.discount**idx) for idx, cost in enumerate(episode_costs)
            ]
            episode_cost_to_go = float(np.sum(discounted_episode_costs))
            last_costs_to_go.append(episode_cost_to_go)

            # Save best agent if the mean over the last episodes improved
            if episode >= avg_cost_to_go_window:
                mean = np.mean(last_costs_to_go)
                if mean < best_avg_cost_to_go:
                    best_avg_cost_to_go = mean
                    self.save_best_weights(episode)

            # Print some info
            if goal_reached:
                print("\t Reached goal!")
            else:
                print("\t Did not reach goal...")
            print(f"\t Epsilon: {epsilon}")
            print(f"\t Cost to go: {episode_cost_to_go}")
            print(f"\t Loss: {float(np.mean(episode_losses))}")
            print(f"\t Elapsed seconds: {episode_time}")

            # Also save the weights every number of episodes
            if episode % 25 == 0:
                self._q_network.save_weights(f"{self._model_folder}/backup_weights_{episode}.h5")

            # Save data
            self.save_data(
                np.array(episodes_costs),
                np.array(episodes_losses),
                float(np.mean(episodes_time))
            )

        self.save_data(
            np.array(episodes_costs),
            np.array(episodes_losses),
            float(np.mean(episodes_time))
        )

        return tf.keras.models.clone_model(self._q_network)

    def choose_action(self, state: npt.NDArray, epsilon: float) -> int:
        """
        Choose an action, either randomly or greedily.

        Args:
            state: the current state.
            epsilon: the probability of taking a random action.

        Returns:
            An action.
        """
        if self._rng.uniform() < epsilon:
            action = self._rng.integers(0, self._env.num_controls)
        else:
            state_tf = tf.convert_to_tensor(state)
            state_tf = tf.expand_dims(state_tf, axis=0)
            q_values = self._q_network(state_tf, training=False)
            q_values = tf.squeeze(q_values)
            action = int(np.argmin(q_values))
        return action

    def training_step(self, optimizer: tf.keras.optimizers.Optimizer) -> float:
        """
        Performs a training step over a batch of experiences.

        Returns:
            The mean loss value (of the batch).
        """
        loss_function = tf.keras.losses.MeanSquaredError()

        with tf.GradientTape() as tape:
            # Sample experiences and convert them into tensors
            start_states, actions, next_states, costs, goals = self._exp_buffer.sample(self._hyper_params.batch_size)

            start_states_tf = tf.convert_to_tensor(start_states, dtype=tf.float32)
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
            next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
            costs_tf = tf.convert_to_tensor(costs, dtype=tf.float32)
            goals_tf = tf.convert_to_tensor(goals, dtype=tf.bool)

            # Compute target values
            q_values_next_states = self._q_target(next_states_tf, training=False)
            min_q_values_next_states = tf.reduce_min(q_values_next_states, axis=1)
            target_state_action_value = tf.where(
                goals_tf,
                costs_tf,
                costs_tf + (self._hyper_params.discount * min_q_values_next_states)
            )

            # Compute actual values
            q_values_start_states = self._q_network(start_states_tf, training=True)
            # We make a 2D matrix of indices to index the Q-value corresponding
            # to each experience's performed action
            row_indices = tf.range(tf.shape(actions_tf)[0], dtype=actions_tf.dtype)
            full_indices = tf.stack([row_indices, actions_tf], axis=1)
            actual_state_action_value = tf.gather_nd(
                q_values_start_states,
                full_indices
            )

            # Compute loss
            loss = loss_function(target_state_action_value, actual_state_action_value)

        # Compute gradients and backpropagate
        gradients = tape.gradient(loss, self._q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self._q_network.trainable_variables))

        # Return mean batch loss
        return loss / len(start_states)

    def save_params(self) -> None:
        model = self._q_network

        params = {
            "input_size": model.input_shape[1],
            "output_size": model.output_shape[1],
            "max_vel": self._env.agent.max_vel,
            "max_torque": self._env.agent.max_torque
        }

        to_save = {
            "env": params,
            "hyper": self._hyper_params._asdict()
        }

        # Save
        with open(f"{self._model_folder}/params.json", "w") as file:
            json.dump(to_save, file)

    def save_best_weights(self, episode: int) -> None:
        # Find previous best weights and delete them if present
        weights = [item for item in os.listdir(self._model_folder) if item.startswith("weights_")]
        if len(weights) != 0:
            os.remove(f"{self._model_folder}/{weights[0]}")
        # Save new best weights
        self._q_network.save_weights(f"{self._model_folder}/weights_{episode}.h5")

    def save_data(self, episodes_costs: npt.NDArray, episodes_losses: npt.NDArray, avg_time: float) -> None:
        with open(f"{self._model_folder}/avg_time.json", "w") as file:
            json.dump(avg_time, file)

        np.save(f"{self._model_folder}/costs.npy", episodes_costs)
        np.save(f"{self._model_folder}/losses.npy", episodes_losses)

    @classmethod
    def load(cls, name: str, weights_name: str = None) -> Tuple[tf.keras.Model, PendulumEnv]:

        model_folder = f"{cls.save_folder}/{name}"
        with open(f"{model_folder}/params.json", "r") as file:
            params = json.load(file)

        params = params["env"]

        model = Network.get_model(params["input_size"], params["output_size"])
        if weights_name is None:
            weights_name = [item for item in os.listdir(model_folder) if item.startswith("weights_")][0]
        model.load_weights(f"{model_folder}/{weights_name}")

        single = params["input_size"] == 2
        if single:
            env = SinglePendulumEnv(
                params["output_size"], params["max_vel"], params["max_torque"]
            )
        else:
            env = UnderactDoublePendulumEnv(
                params["output_size"], params["max_vel"], params["max_torque"]
            )

        return model, env
