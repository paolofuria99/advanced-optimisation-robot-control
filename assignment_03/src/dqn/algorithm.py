from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import NamedTuple, Tuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.pendulum as environment
import tensorflow as tf
from orc.assignment_03.src.dqn.experience import Experience, ExperienceBuffer
from orc.assignment_03.src.dqn.network import Network
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
            env: environment.Pendulum,
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
            The Q-network from the last episode. The best model is saved separately..
        """
        np_config.enable_numpy_behavior()

        # Keep track of best model to return
        best_model = tf.keras.models.clone_model(self._q_network)

        # Set epsilon to its starting value, which will be decayed
        epsilon = self._hyper_params.epsilon_start

        # Keep track of the total number of steps (regardless of the episode)
        total_steps = 0

        # Keep track of average episode time and costs of each episode
        episodes_time = []
        episodes_costs = []

        # cost_to_go mean every tot
        cost_to_go_all = []
        mean_every = 5
        cost_to_go_best_mean = np.inf

        # Run training for a maximum number of episodes
        for episode in range(1, self._hyper_params.max_episodes):

            print("======================")
            print(f"EPISODE {episode}")
            print("======================")

            # Flag to know if to display or not
            display = (episode % self._hyper_params.display_every_episodes) == 0

            # Initialize variables to keep track of progress
            goal_reached = False
            episode_costs = []

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
                if total_steps >= self._hyper_params.replay_start:
                    self.training_step()

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

            # Keep track of episode costs
            episodes_costs.append(episode_costs)

            # Compute cost to go
            discounted_episode_costs = [
                cost * (self._hyper_params.discount**idx) for idx, cost in enumerate(episode_costs)
            ]
            episode_cost_to_go = float(np.sum(discounted_episode_costs))

            # Inserting the cost to go
            cost_to_go_all.append(episode_cost_to_go)

            # Save best model if the mean over the last episodes improved
            if episode >= mean_every:
                mean = np.mean(cost_to_go_all[episode - mean_every:episode])
                if mean < cost_to_go_best_mean:
                    cost_to_go_best_mean = mean
                    self.save_best_weights(episode)

            # Print some info
            if goal_reached:
                print("\t Reached goal!")
            else:
                print("\t Did not reach goal...")
            print(f"\t Epsilon: {epsilon}")
            print(f"\t Cost to go: {episode_cost_to_go}")
            print(f"\t Elapsed seconds: {episode_time}")

        self.save_costs_and_avg_time(
            np.array(episodes_costs),
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

    def training_step(self) -> float:
        """
        Performs a training step over a batch of experiences.

        Returns:
            The mean loss value (of the batch).
        """
        loss_function = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(self._hyper_params.learning_rate)

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
            "time_step": self._env.time_step,
            "max_vel": self._env.max_vel,
            "max_torque": self._env.max_torque
        }

        # Save parameters
        with open(f"{self._model_folder}/params.json", "w") as file:
            json.dump(params, file)

        # Save hyper parameters
        with open(f"{self._model_folder}/hyper.json", "w") as file:
            json.dump(self._hyper_params._asdict(), file)

    def save_best_weights(self, episode: int) -> None:
        # Find previous best weights and delete them if present
        weights = [item for item in os.listdir(self._model_folder) if item.startswith("weights_")]
        if len(weights) != 0:
            os.remove(f"{self._model_folder}/{weights[0]}")
        # Save new best weights
        self._q_network.save_weights(f"{self._model_folder}/weights_{episode}.h5")

    def save_costs_and_avg_time(self, episodes_costs: npt.NDArray, avg_time: float) -> None:
        with open(f"{self._model_folder}/avg_time.json", "w") as file:
            json.dump(avg_time, file)

        np.save(f"{self._model_folder}/costs.npy", episodes_costs)

    @classmethod
    def load(cls, name: str) -> Tuple[tf.keras.Model, environment.Pendulum]:

        model_folder = f"{cls.save_folder}/{name}"
        with open(f"{model_folder}/params.json", "r") as file:
            params = json.load(file)

        model = Network.get_model(params["input_size"], params["output_size"])
        weights = [item for item in os.listdir(model_folder) if item.startswith("weights_")][0]
        model.load_weights(f"{model_folder}/{weights}")

        single = params["input_size"] == 2
        if single:
            env = environment.SinglePendulum(
                params["time_step"], params["output_size"], params["max_vel"], params["max_torque"]
            )
        else:
            env = environment.DoublePendulumUnderact(
                params["time_step"], params["output_size"], params["max_vel"], params["max_torque"]
            )

        return model, env
