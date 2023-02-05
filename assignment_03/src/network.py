from __future__ import annotations

from typing import Tuple, NamedTuple

import numpy as np
import numpy.typing as npt
import orc.assignment_03.src.environment.pendulum as environment
import tensorflow as tf
from orc.assignment_03.src.utils import Utils
from tensorflow.python.ops.numpy_ops import np_config


class Network:
    """
    This is a static utility class used to get the network model to be used
    in the Deep Q Learning algorithm.
    """

    @staticmethod
    def get_model(input_size: int, output_size: int, name: str) -> tf.keras.Model:
        """
        Get the neural network model to use for Deep Q Learning.

        Args:
            input_size: the number of neurons in the input layer
            output_size: the number of neurons in the output layer
            name: the name of the model (for logging purposes)

        Returns:
            A neural network model.
        """
        inputs = tf.keras.layers.Input(input_size)
        state_out1 = tf.keras.layers.Dense(16, activation="relu", kernel_initializer="variance_scaling")(inputs)
        state_out2 = tf.keras.layers.Dense(32, activation="relu", kernel_initializer="variance_scaling")(state_out1)
        state_out3 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="variance_scaling")(state_out2)
        state_out4 = tf.keras.layers.Dense(64, activation="relu", kernel_initializer="variance_scaling")(state_out3)
        outputs = tf.keras.layers.Dense(output_size)(state_out4)

        return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class DQNet:
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

    class Experience(NamedTuple):
        """
        This class describes an experience to be stored in the experience-replay buffer.
        """
        curr_state: npt.NDArray
        action: int
        next_state: npt.NDArray
        cost: float
        is_goal: bool

    class ExperienceBuffer:
        """
        This class implements an efficient circular buffer for the experience-replay buffer.
        We make a custom class and not use something like collections.deque because we want
        both efficient circular appends (with a maximum size) and efficient random indexing,
        and the latter is not provided by deque.
        """

        def __init__(self, size: int, rng: np.random.Generator) -> None:
            self._buffer = np.empty(size, dtype=DQNet.Experience)
            self._index = 0
            self._num_elements = 0
            self._size = size
            self._rng = rng

        def append(self, experience: DQNet.Experience) -> None:
            """
            Append an experience to the buffer. If the buffer is full,
            the oldest experience is replaced.

            Args:
                experience: the experience to add.
            """
            self._buffer[self._index] = experience
            self._index = (self._index + 1) % self._size
            if self._num_elements < self._size:
                self._num_elements += 1

        def sample(self, quantity: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
            """
            Sample uniformly a batch of experiences.

            Args:
                quantity: how many experiences to sample.

            Returns:
                A tuple containing, respectively,
                an array of all the starting states of each experience;
                an array of all the actions taken in each experience;
                an array of all the resulting states of each experience (after applying the corresponding action);
                an array of all the costs of taking an action in each experience;
                a boolean mask that says which experiences resulted in a goal state.
            """
            indices = self._rng.choice(self._num_elements, quantity, replace=False)
            curr_states, actions, next_states, costs, goals = zip(*[self._buffer[idx] for idx in indices])
            return np.array(curr_states), np.array(actions), np.array(next_states), np.array(costs), np.array(goals)

        def is_full(self) -> bool:
            """
            Check if the buffer is full or not.

            Returns:
                True if it is full, False otherwise.
            """
            return self._num_elements == self._size

        def has_at_least(self, quantity: int) -> bool:
            """
            Check if the buffer has at least a certain amount of experiences.

            Args:
                quantity: the minimum number of experiences wanted in the buffer.

            Returns:
                True if there are enough experiences, False otherwise.
            """
            return self._num_elements >= quantity

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
        self._exp_buffer = DQNet.ExperienceBuffer(self._hyper_params.replay_size, self._rng)

    def train(self) -> tf.keras.Model:
        """
        Train the Deep Q network.

        Returns:
            The trained Q-network.
        """
        np_config.enable_numpy_behavior()

        # Keep track of the best model
        best_model = None
        best_model_loss = np.inf

        # Set epsilon to its starting value, which will be decayed
        epsilon = self._hyper_params.epsilon_start

        try:
            # Run training for a maximum number of episodes
            for episode in range(self._hyper_params.max_episodes):

                print("======================")
                print(f"EPISODE {episode + 1}")
                print("======================")

                display = ((episode + 1) % self._hyper_params.display_every_episodes) == 0
                reached_goal = False

                # Reset the environment to a random state
                self._env.reset(display=display)
                # Keep track of running loss to compute mean loss for each episode
                running_loss = 0.

                # Run each episode for a maximum number of steps (or until the state is terminal)
                for step in range(self._hyper_params.max_steps_per_episode):

                    # Decay the epsilon
                    epsilon = max(epsilon * self._hyper_params.epsilon_decay, self._hyper_params.epsilon_min)

                    # Choose an action and perform a step from the current state
                    curr_state = self._env.current_state
                    action = self.choose_action(epsilon)
                    new_state, cost = self._env.step(action, display=display)

                    # Store the experience in the experience-replay buffer
                    self._exp_buffer.append(
                        DQNet.Experience(curr_state.to_np(), action, new_state.to_np(), cost, new_state.is_goal())
                    )

                    # Perform a training step (if the experience buffer has enough elements)
                    if self._exp_buffer.has_at_least(self._hyper_params.batch_size):
                        running_loss += self.training_step()

                    # Copy weights to target network every number of steps
                    if step != 0 and (step + 1) % self._hyper_params.steps_for_target_update == 0:
                        print(f"\t Step {step + 1}, copying weights to target network.")
                        self._q_target.set_weights(self._q_network.get_weights())

                    # End the episode if reached the goal
                    if new_state.is_goal():
                        reached_goal = True
                        break

                if reached_goal:
                    print("\t Reached goal!")
                    mean_loss = running_loss / (episode + 1)
                    print(f"\t Loss: {mean_loss}")
                    print(f"\t Epsilon: {epsilon}")

                    # Save the model as best if loss is better
                    if best_model is None or mean_loss < best_model_loss:
                        best_model = tf.keras.models.clone_model(self._q_network)
                        best_model_loss = mean_loss
                else:
                    print("\t Did not reach goal...")
                    mean_loss = running_loss / self._hyper_params.max_steps_per_episode
                    print(f"\t Loss: {mean_loss}")
                    print(f"\t Epsilon: {epsilon}")
        except KeyboardInterrupt:
            print("INTERRUPTED!")
            return best_model
        except Exception as e:
            raise e

    def choose_action(self, epsilon: float) -> int:
        """
        Choose an action, either randomly or greedily.

        Args:
            epsilon: the probability of taking a random action

        Returns:
            An action.
        """
        if self._rng.uniform() < epsilon:
            action = self._rng.integers(0, self._env.num_controls)
        else:
            np_state = self._env.current_state.to_np().reshape((1, -1))
            action = int(np.argmin(self._q_network(np_state), axis=1)[0])
        return action

    def training_step(self) -> float:
        """
        Performs a training step over a batch of experiences.

        Returns:
            The loss value.
        """
        loss_function = tf.keras.losses.MeanSquaredError()
        optimizer = tf.keras.optimizers.Adam(self._hyper_params.learning_rate)

        with tf.GradientTape() as tape:
            # Sample experiences, unpack them, and convert them into tensors
            curr_states, actions, next_states, costs, goals = self._exp_buffer.sample(self._hyper_params.batch_size)
            curr_states = Utils.np_2_tf(curr_states)
            actions = Utils.np_2_tf(actions)
            next_states = Utils.np_2_tf(next_states)
            costs = Utils.np_2_tf(costs)
            goals = Utils.np_2_tf(tf.squeeze(goals))

            # Compute target values
            # The target network is not to be trained
            target_state_action_value = tf.reduce_min(self._q_target(next_states, training=False), axis=1)
            # If the state is a goal, keep only the cost; otherwise, keep the Q-value
            target_state_action_value = tf.where(
                goals, tf.cast(costs, target_state_action_value.dtype), target_state_action_value
            )

            # Compute actual Q values
            # We make a 2D matrix of indices to index the result of calling the Q-network and get the
            # Q-value corresponding to each experience's performed action
            row_indices = tf.range(tf.shape(actions)[0], dtype=actions.dtype)
            full_indices = tf.stack([row_indices, actions], axis=1)
            actual_state_action_value = tf.gather_nd(
                self._q_network(curr_states, training=True),
                full_indices
            )

            # Compute loss
            loss = loss_function(target_state_action_value, actual_state_action_value)

        # Compute gradients and backpropagate
        gradients = tape.gradient(loss, self._q_network.trainable_weights)
        optimizer.apply_gradients(zip(gradients, self._q_network.trainable_weights))

        return loss
