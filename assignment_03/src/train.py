import argparse
from datetime import datetime
from enum import Enum

import numpy.random as random
import orc.assignment_03.src.environment.pendulum as environment
from orc.assignment_03.src.dqn.algorithm import DQL
from orc.assignment_03.src.dqn.network import Network


class PendulumType(Enum):
    SINGLE = "single"
    DOUBLE = "double"


def main(
        hyper_params: DQL.HyperParams,
        name: str,
        num_controls: int = 11,
        pend_type: PendulumType = PendulumType.SINGLE,
        rng_seed: int = None
):
    if rng_seed is not None:
        rng = random.default_rng(seed=42)
    else:
        rng = random.default_rng()

    if pend_type == pend_type.SINGLE:
        num_joints = 1

        env = environment.SinglePendulum(max_vel=8.0, max_torque=2.0, rng=rng)
        model = Network.get_model(num_joints * 2, num_controls, name)

        dql = DQL(model, hyper_params, env, rng=rng)
    else:
        print("Not supported")
        return

    trained_model = dql.train()

    time = datetime.now().strftime("%H%M%S")
    trained_model.save_weights(f"models/{name}_{time}.h5")


if __name__ == "__main__":
    possible_types = [str(x.value) for x in PendulumType]

    parser = argparse.ArgumentParser()

    parser.add_argument("--type", required=True, help="the type of pendulum to use", type=str, choices=possible_types)
    parser.add_argument("--name", required=True, help="the name of the experiment", type=str)
    parser.add_argument("--controls", required=False, help="how many controls to use", type=int, default=11)
    parser.add_argument("--seed", required=False, help="the rng seed", type=int)

    args = parser.parse_args()

    hp = DQL.HyperParams(
        replay_size=10000,
        replay_start=2500,
        discount=0.99,
        max_episodes=200,
        max_steps_per_episode=500,
        steps_for_target_update=1000,
        epsilon_start=1.0,
        epsilon_decay=0.985,
        epsilon_min=0.002,
        batch_size=32,
        learning_rate=0.001,
        display_every_episodes=10
    )

    main(
        hp,
        name=args.name,
        num_controls=args.controls,
        pend_type=PendulumType(args.type),
        rng_seed=args.seed
    )
