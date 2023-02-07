from datetime import datetime

import orc.assignment_03.src.environment.pendulum as environment
from orc.assignment_03.src.dqn.algorithm import DQL
from orc.assignment_03.src.dqn.network import Network

import numpy.random as random


def main():
    hyper_params = DQL.HyperParams(
        replay_size=10000,
        replay_start=32,
        discount=0.99,
        max_episodes=1,
        max_steps_per_episode=500,
        steps_for_target_update=1000,
        epsilon_start=1.0,
        epsilon_decay=0.9985,
        epsilon_min=0.002,
        batch_size=32,
        learning_rate=0.001,
        display_every_episodes=10
    )

    num_joints = 1
    num_controls = 11
    name = "single"

    rng = random.default_rng(seed=42)

    env = environment.SinglePendulum(time_step=0.005, rng=rng)
    model = Network.get_model(num_joints*2, num_controls, name)

    dql = DQL(model, hyper_params, env, rng=rng)

    best_model = dql.train()

    if best_model is not None:
        time = datetime.now().strftime("%H%M%S")
        best_model.save_weights(f"models/{name}_{time}.h5")


if __name__ == "__main__":
    main()
