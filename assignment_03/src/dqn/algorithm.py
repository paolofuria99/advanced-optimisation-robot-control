from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.pendulum as environment
import tensorflow as tf
from orc.assignment_03.src.dqn.experience import Experience, ExperienceBuffer
from tensorflow.python.ops.numpy_ops import np_config


class DQL:
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
            network_model: tf.keras.Model,
            hyper_params: HyperParams,
            env: environment.Pendulum,
            rng: np.random.Generator = np.random.default_rng()
    ) -> None:
        self._q_network = tf.keras.models.clone_model(network_model)
        self._q_target = tf.keras.models.clone_model(network_model)
        self._hyper_params = hyper_params
        self._env = env

        # Set the rng
        self._rng = rng

        # Prepare the experience replay buffer
        self._exp_buffer = ExperienceBuffer(self._hyper_params.replay_size, self._rng)

    def train(self) -> tf.keras.Model:
        """
        Train the Q-network.

        Returns:
            The best performing Q-network.
        """
        np_config.enable_numpy_behavior()

        # Keep track of best model to return
        best_model = tf.keras.models.clone_model(self._q_network)

        # Set epsilon to its starting value, which will be decayed
        epsilon = self._hyper_params.epsilon_start

        # Keep track of the total number of steps (regardless of the episode)
        total_steps = 0

        # Run training for a maximum number of episodes
        for episode in range(self._hyper_params.max_episodes):

            print("======================")
            print(f"EPISODE {episode + 1}")
            print("======================")

            # Flag to know if to display or not
            display = ((episode + 1) % self._hyper_params.display_every_episodes) == 0

            # Initialize variables to keep track of progress
            goal_reached = False
            episode_cost = 0.0

            # Set the environment to a random state
            self._env.reset(display=display)

            # Run each episode for a maximum number of steps (or until the state is terminal)
            for step in range(self._hyper_params.max_steps_per_episode):

                total_steps += 1

                # Decay the epsilon
                epsilon = max(epsilon * self._hyper_params.epsilon_decay, self._hyper_params.epsilon_min)

                # Choose an action and perform a step from the current state
                curr_state = self._env.current_state
                action = self.choose_action(curr_state, epsilon)
                new_state, cost, goal_reached = self._env.step(action, display=display)

                episode_cost += cost

                # Store the experience in the experience-replay buffer
                self._exp_buffer.append(
                    Experience(curr_state, action, new_state, cost, goal_reached)
                )

                # Perform a training step if enough steps have been performed
                if total_steps % self._hyper_params.replay_start == 0:
                    self.training_step()

                # Copy weights to target network every certain number of steps
                if total_steps % self._hyper_params.steps_for_target_update == 0:
                    print(f"\t Copying weights to target network.")
                    self._q_target.set_weights(self._q_network.get_weights())

                # End the episode if reached the goal
                if goal_reached:
                    break

            if goal_reached:
                print("\t Reached goal!")
            else:
                print("\t Did not reach goal...")
            print(f"\t Epsilon: {epsilon}")
            print(f"\t Episode cost: {episode_cost}")

            if ((episode + 1) % 30) == 0:
                best_model = tf.keras.models.clone_model(self._q_network)

        return best_model

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
