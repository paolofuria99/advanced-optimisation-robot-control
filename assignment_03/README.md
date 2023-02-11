# Deep Q-Learning for pendulum swing-up

This folder contains all the code that implements a Deep Q-Learning training and evaluation setting for two agents: a single pendulum, and an underactuated double pendulum.

## Requirements
- The robotics environment provided during the _Optimization Based Robot Control_ course @ University of Trento.
- Tensorflow.

## Project structure
```
results/                        [folder with the results of training and evaluation]
├─ double/                      [results of double pendulum]
├─ single/                      [results of single pendulum]
src/                            [folder containing all source code]
├─ agent/                       [package implementing the RL agents]
│  ├─ double/                   [package implementing the double pendulum agent]
│  │  ├─ double_pendulum.py     [implementation of double pendulum agent]
│  │  ├─ simulator.py           [code to connect to Gepetto viewer]
│  │  ├─ wrapper.py             [code to load and wrap the model]
│  ├─ single/                   [package implementing the single pendulum agent]
│  │  ├─ display.py             [code to connect to Gepetto viewer]
│  │  ├─ single_pendulum        [implementation of single pendulum agent]
│  ├─ utils.py                  [utility functions for the agents]
│  ├─ pendulum.py               [abstract class for a pendulum agent]
├─ dqn/                         [package implementing the DQL algorithm]
│  ├─ algorithm.py              [the main algorithm of DQL]
│  ├─ experience.py             [implementation of DQL experience buffer]
│  ├─ network.py                [implementation of the Q network]
├─ environment/                 [package implementing the RL environments]
│  ├─ double_pendulum.py        [implementation of under-actuated double pendulum environment]
│  ├─ pendulum.py               [abstract class for a pendulum environment]
│  ├─ single_pendulum.py        [implementation of single pendulum environment
├─ models/                      [folder saving data of training]
│  ├─ double_best/              [folder containing data of the best double pendulum model]
│  ├─ single_best/              [folder containing data of the best sigle pendulum model]
├─ eval.py                      [main entry-point for evaluating a model]
├─ plot.py                      [code implementing plotting functions]
├─ train.py                     [main entry-point for training a model]
README.md                       
```
## Usage
There are two main entrypoints: `training.py` for training a model, and `eval.py` for evaluating a trained model. In order to run them, you need to move into the `src` folder.

### Training
To start training a model with default parameters, for the single or double pendulum respectively, run:
```
python3 train.py [single|double] name_of_experiment
```
Optionally, you can specify the following parameters:


| Name and value                | Description                                                 | Default  |
|-------------------------------|-------------------------------------------------------------|----------|
| `--controls CONTROLS`         | number of controls available (discretizes the torque range) | 11       |
| `--max-vel MAX_VEL`           | the maximum velocity of the pendulum joints                 | 5.0      |
| `--max-torque MAX_TORQUE`     | the maximum torque of the pendulum actuators                | 5.0      |
| `--seed SEED`                 | the rng seed for everything random in the algorithm         | *random* |
| `--replay-size REPLAY_SIZE`   | the experience replay buffer size                           | 10000    |
| `--replay-start REPLAY_START` | how many steps before replay training starts                | 1000     |
| `--discount DISCOUNT`         | the discount factor                                         | 0.99     |
| `--max-episodes MAX_EPISODES` | the maximum number of episodes                              | 100      |
| `--max-steps MAX_STEPS`       | the maximum number of steps per episode                     | 500      |
| `--sync-target SYNC_TARGET`   | how often (steps) to update target network                  | 1000     |
| `--eps-start EPS_START`       | the starting value of epsilon                               | 1.0      |
| `--eps-decay EPS_DECAY`       | the decay of epsilon (eps=eps*decay)                        | 0.995    |
| `--eps-min EPS_MIN`           | the minimum value of epsilon                                | 0.005    |
| `--batch-size BATCH_SIZE`     | the size of an experience batch                             | 128      |
| `--lr LR`                     | the initial learning rate                                   | 0.001    |
| `--display-rate DISPLAY_RATE` | how often (episodes) to display in Gepetto                  | 10       |

The results of training are stored in `models/name_of_experiment`, together with the parameters. During training, the best model is saved at each episode, and additional checkpoints are saved every 25 episodes.

### Evaluation
To evaluate a trained model, run:
```
python3 eval.py name_of_the_experiment
```
where `name_of_the_experiment` is the name you have chosen for training. Since the evaluation automatically loads the right environment and model architecture for either the single or double pendulum, you need to have this folder structure:
```
models/
├─ name_of_experiment/    [the experiment you want to evaluate]
│  ├─ best_weights_XX.h5  [the best model weights, XX is the episode at which they are saved]
│  ├─ params.json         [contains all the parameters of the experiment]
```
Optionally, you can specify the following parameters:

| Name and value                | Description                              | Default            |
|-------------------------------|------------------------------------------|--------------------|
| `--weights WEIGHTS`           | the name of the weights to load          | best_weights_XX.h5 |
| `--num-episodes NUM_EPISODES` | the number of episodes to run            | 1                  |
| `--duration DURATION`         | duration of an episode, in seconds       | 7.0                |
| `-r`                          | if set, start episode in random position | False              |
| `-np`                         | if set, do not plot results              | False              |

Two models are provided and ready for evaluation: the best-performing models we found for the single and double pendulum. You can run their evaluation like so:
```
python3 eval.py single_best
```
```
python3 eval.py double_best
```

## Results of our experiments
Plots, images and videos of the best models we found for both pendulums can be found in the `results` folder. For more details, refer to our report.

Below you can find videos of the simulations of both pendulums, both starting from the bottom and starting from 10 different random states.

https://user-images.githubusercontent.com/50495055/218241298-587604c8-bb9b-46f1-99c5-123df4337460.mp4

https://user-images.githubusercontent.com/50495055/218241319-ccea4300-86fc-45fa-88d4-4f2e30e7f398.mp4

https://user-images.githubusercontent.com/50495055/218241340-4ed5603c-2462-4042-b4ba-52fe65a9e558.mp4

https://user-images.githubusercontent.com/50495055/218241368-29fe2117-a27d-4ea8-a54a-0fbc74101e7d.mp4
