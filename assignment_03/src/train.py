import orc.assignment_03.src.environment.pendulum as environment
from orc.assignment_03.src.network import DQNet, Network

import numpy.random as random


def main():
    hyper_params = DQNet.HyperParams(
        replay_size=10000,
        replay_start=1000,
        discount=0.99,
        max_episodes=500,
        max_steps_per_episode=100,
        steps_for_target_update=50,
        epsilon_start=1.,
        epsilon_decay=0.99,
        epsilon_min=0.01,
        batch_size=256,
        learning_rate=0.001,
        display_every_episodes=25
    )

    num_joints = 1
    num_controls = 32
    name = "single"

    rng = random.default_rng(seed=42)

    env = environment.SinglePendulum(num_controls=num_controls, max_torque=10., rng=rng)
    model = Network.get_model(num_joints*2, num_controls, name)

    dq = DQNet(model, hyper_params, env, rng=rng)

    best_model = dq.train()

    if best_model is not None:
        best_model.save(f"models/{name}")


if __name__ == "__main__":
    main()
