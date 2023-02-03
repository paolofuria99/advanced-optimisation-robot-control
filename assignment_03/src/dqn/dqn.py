from src.environment.pendulumH import SinglePendulum

import tensorflow as tf
from keras import layers
import numpy as np
from numpy.random import randint, uniform
import matplotlib.pyplot as plt
import time
from collections import deque
import sys
import hyper as h
from random import sample
import pandas as pd
import keyboard

from tensorflow.python.ops.numpy_ops import np_config

# Tensorflow conversions stuff
def np2tf(y):
    """
    convert from numpy to tensorflow
    """
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out

def tf2np(y):
    """
    convert from tensorflow to numpy
    """
    return tf.squeeze(y).numpy()



class DeepQNN:
    def __init__(
            self,
            hidden_layers = 3,
            number_of_joints = 1
    ):
        """
        This class depicts a Deep Q network applied to a pendulum with a variable number of joints

        Args:
            #control_size: number of discretization steps for joint torque
            hidden_layers: number of hidden layers
            number_of_joints: number of joints
        """

        # Variables useful
        self.printF = 25  # Printing frequency
        self.export = True  # export cost-to-go
        self.plotFinal = True  # plot final cost-to-go
        self.debug = True  # Uncheck for debug

        # Initialize the DQNN
        self.hidden_layers = hidden_layers
        self.number_of_joints = number_of_joints

        if number_of_joints == 1:
            self.env = SinglePendulum()
            self.control_size = self.env.control_size
            self._state_dimensions = self.env._state_dimensions
            if self.debug: print("Single Pendulum initilized")
        elif number_of_joints == 2:
            #self.env = DoublePendulum()
            #if self.debug: print("Double Pendulum initilized")
            #todo: implement double pendulum
            pass
        else: sys.exit("Number of joints is unvailable")

        if hidden_layers == 3:
            self.NN_layers =3
            self.q = self.get_critic3()
            self.q.summary()
            self.q_target = self.get_critic3()
            self.q_target.set_weigths(self.q.get_weights())
            if self.debug: print("Neural network initilized correctly")
        else:
            pass
            # todo: maybe if we implement NN with different hidden layers

        # Set-up optimizer
        self.optimizer = tf.keras.optimizers.Adam(h.QVALUE_LEARNING_RATE)  # Stochastic gradient descent SGD

        # Initialize replay buffer-size
        self.replay_buffer = deque(maxlen=h.REPLAY_BUFFER_SIZE)

        # Initialize cost-2-go
        self.c2gRecord = []  # Storing cost-to-go
        self.c2gBest = np.inf
        self.c2gImprov = []  # storing only the best cost-to-go
        self.c2gEpisode = []  #storing the episode number ar wich the SGD find the better minima

    def update(self, batch):
        """
        Update the weights of the Q network using the specified batch of data

        Args:
            batch: array of where each component is composed of [state, action, reward, next_state, final_state]
        """
        n = len(batch)  # length of batch

        x_batch = np.array([sample[0] for sample in batch])
        u_batch = np.array([sample[1] for sample in batch])
        cost_batch = np.array([sample[2] for sample in batch])
        x_next_batch = np.array([sample[3] for sample in batch])
        finished_batch = np.array([sample[4] for sample in batch])

        # all inputs are tf tensors
        with tf.GradientTape() as tape:
            # Compute Q target using NN
            target_value = self.q_target(x_next_batch, training=True)

            # Compute 1-step targets for the critic loss
            y = np.zeros(n)  # initialize
            for id, finished in enumerate(finished_batch):
                if finished:
                    y[id] = cost_batch[id]
                else:
                    y[id] = cost_batch[id] + h.GAMMA*target_value[id]

            # Compute batch of values associated to the sampled batch of states
            Q_Value = self.q(x_batch, training = True)
            Q_Loss = tf.math.reduce_mean(tf.math.square(y-Q_Value))

            # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
            Q_Grad = tape.gradient(Q_Loss, self.q.trainable_variables)

            # Update the critic backpropagating the gradients
            self.optimizer.apply_gradients((zip(Q_Grad, self.q.trainable_variables)))

    def chooseU(self, x, epsilon):
        """
        Choose the discrete control of the system following an epsilon greedy strategy

        Args:
            x:
            epsilon:
        """
        if uniform(0, 1) < epsilon:  # explore
            u = randint(self.control_size, size=self.number_of_joints)
        else:  # exploit
            pred = self.q.predict(x.reshape(1, -1))
            u = np.argmin(pred.reshape(self.control_size, self.number_of_joints), axis=0)

        if len(u) == 1:
            u = u[0]
        return u

    def trainNN(self):
        """
        Training the NN
        """

        steps = 0
        epsilon = h.EPSILON  # import epsilon

        t = time.time()  # start measuring the time needed for training

        for episode in range(h.EPISODES):
            c2go = 0.0
            x = self.env.reset()
            gamma = 1
            self.episodeExport = episode  # keep track of current episode for plotting on-the-go and exporting data to external .csv file

            for step in range(h.MAX_EPISODE_LENGTH):  # SAMPLING PHASE
                u = self.chooseU(x, epsilon)
                x_next, cost = self.env.step(u)

                finished = True if step == h.MAX_EPISODE_LENGTH - 1 else False

                # # REPLAY MEMORY -- Break correlation between consecutive samples, if network would learn only from them and they're highly correlated samples
                # thus inefficient learning

                self.replay_buffer.append(
                    [x, u, cost, x_next, finished])  # agent experience at step t stored in replay memory

                # update weights of target network according to hyperparameters
                if steps % h.UPDATE_Q_TARGET_STEPS == 0:
                    self.q_target.set_weights(self.q.get_weights())

                # Sampling from replay buffer and train NN accordingly
                if len(self.replay_buffer) > h.MIN_BUFFER_SIZE and steps % h.SAMPLING_STEPS == 0:  # TRAINING PHASE
                    batch = sample(self.replay_buffer, h.BATCH_SIZE)
                    self.update(batch)

                # update state , steps and discount factor accordingly
                x = x_next
                c2go += gamma * cost
                gamma *= h.GAMMA
                steps += 1

            # Save NN weights everytime a better cost to go is found and plot the average cost to go vs the best
            if abs(c2go) < abs(self.Bestc2go):
                self.saveModel()
                self.Bestc2go = c2go  # update best cost to go
                self.c2gImprov.append(self.Bestc2go)  # store best cost to go
                self.c2gEpisode.append(episode)  # store at what episode it was found
                self.plotting(episode)

            epsilon = max(h.MIN_EPSILON, np.exp(-h.EPSILON_DECAY * episode))  # calculate the decay of epsilon
            self.c2gRecord.append(c2go)  # append current cost to go in array

            # Regularly print in terminal useful info about how the training is proceding
            if episode != 0 and episode % self.printF == 0:
                dt = time.time() - t  # calculate elapsed time since last printing to terminal
                t = time.time()

                if episode % 50 == 0:  # save model every 50 episodes
                    print(50 * "--")
                    print("saving and plotting model at episode", episode)
                    self.saveModel()
                    # self.plotting(self.episodeExport)

                print(50 * "--")
                print(
                    'episode %d | cost %.1f | exploration prob epsilon %.6f | time elapase [s] %.5f s | cost to go improved in total %d times | best cost to go %.3f' % (
                        episode, 0, epsilon, dt, len(self.c2gImprov),
                        self.Bestc2go))
                print(50 * "--")
                self.plotting(self.episodeExport + 1)

        if self.plotFinal:
            self.plotting(h.EPISODES)
            self.exportCosts(h.EPISODES)

    def plotting(self, episodes):
        ''' Plot the average cost-to-go history and best cost to go and its relative episode of when it was found  '''
        plt.plot(np.cumsum(self.c2gRecord) / range(1, episodes + 1), color='blue')
        plt.scatter(self.c2gEpisode, self.c2gImprov, color='red')
        plt.grid(True)
        plt.xlabel("episodes [n]")
        plt.ylabel("cost to go ")
        plt.legend(["Avg", "Best"])
        plt.title("Average cost-to-go vs Best cost to go update")
        plt.savefig("costToGo.eps")
        # plt.show()
        # time.sleep(2)
        plt.close()

        # episodecost = np.cumsum(self.ctgRecord)/range(1,int(episodes+1))
        cost = pd.Series(self.c2gEpisode, self.c2gImprov)
        cost.to_csv('costsImprov.csv', header=None)

    def saveModel(self):
        print(50 * "#", "New better cost to go found! Saving Model",
              50 * "#")  # the model is also saved at regular intervals
        t = time.time()
        self.q.save_weights(str(t) + "DeepQNN.h5")

    def visualize(self, file_name, x=None):
        """
        Visualize NN results loading model weights and letting it run for 33% of the training episodes

        Args:
            file_name:
        """
        # Load NN weights from file
        self.q.load_weights(file_name)  # load weights

        if x is None:
            x0 = x = self.env.reset()
        else:
            x0 = x

        costToGo = 0.0
        gamma = 1

        for i in range(int(h.EPISODES / 3)):
            pred = self.q.predict(x.reshape(1, -1))  # greedy control selection
            u = np.argmin(pred.reshape(self.control_size, self.number_of_joints), axis=0)
            if len(u) == 1:
                u = u[0]
            x, cost = self.env.step(u)
            costToGo += gamma * cost
            self.c2gRecord.append(costToGo)

            gamma *= h.GAMMA
            self.env.render()

    def exportCosts(self, episodes):
        """
        function used to export the costs for further data analysis in an external file

        Args:
            episodes:
        """
        cost = np.cumsum(self.c2gRecord) / range(1, int(episodes + 1))
        cost = pd.Series(cost)
        cost.to_csv('costs.csv', header=None)

    def get_critic3(self):
        """
        Create the neural network with 3 hidden layers to represent the Q function

        Args:
            nx
        """
        inputs = layers.Input(shape=(self._state_dimensions, 1))
        state_out1 = layers.Dense(64, activation="relu")(inputs)
        state_out2 = layers.Dense(64, activation="relu")(state_out1)
        state_out3 = layers.Dense(64, activation="relu")(state_out2)
        outputs = layers.Dense(1)(state_out3)
        model = tf.keras.Model(inputs, outputs)

        return model


if __name__ == '__main__':

    training = True
    file_name = "../models/Best models/DeepQNN3LayersSingle.h5"  # "models/Best models/DeepQNN3LayersSingle.h5"        # single pendulum model
    # file_name =  "../models/Best models/DeepQNNDouble3.h5" #  Double pendulum model

    deepQN = DeepQNN()  # Input Param: hidden layers , pendulum joints
    if training:
        print(50 * "#")
        print("Beginning training ")
        deepQN.trainNN()

        if keyboard.is_pressed(
                's'):  # manually stop training by pressing S so that the meaningful data of the model are safely saved
            deepQN.saveModel()
            deepQN.exportCosts(deepQN.episodeExport)
            deepQN.plotting(deepQN.episodeExport)
            exit()

    else:
        deepQN.visualize(file_name)  # greedy strategy renderization
