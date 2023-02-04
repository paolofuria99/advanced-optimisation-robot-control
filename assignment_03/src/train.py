from orc.assignment_03.src.network import DQNet, Network
from orc.assignment_03.src.environment.pendulum import SinglePendulum


def main():
    hyper_params = DQNet.HyperParams(
        replay_size=10000,
        replay_start=2000,
        discount=0.99,
        max_episodes=100,
        max_steps_per_episode=10000,
        steps_for_target_update=1000,
        epsilon_start=1.,
        epsilon_decay=0.99995,
        epsilon_min=0.1,
        batch_size=1024,
        learning_rate=0.0001
    )

    num_controls = 11
    display = False
    env = SinglePendulum(num_controls=num_controls, display=display)

    model = Network.get_model(2, num_controls, "trial")

    exit(1)

    dq = DQNet(model, hyper_params, env)
    model = dq.train()
    model.save("models/")


if __name__ == "__main__":
    main()
