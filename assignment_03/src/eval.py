import argparse
import os

from dqn.algorithm import DQL


def main(
        name: str,
        weights: str,
        random_start: bool,
        num_episodes: int,
        max_steps: int
) -> None:
    network, env = DQL.load(name, weights_name=weights)
    env.render_greedy_policy(
        network,
        random_start=random_start,
        max_steps=max_steps,
        num_episodes=num_episodes,
        display=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", help="the name of the agent to evaluate", type=str)
    parser.add_argument("--weights", required=False, help="the name of the weights to load", type=str)
    parser.add_argument("--max-steps", required=False, help="max number of steps per episode", type=int)
    parser.add_argument("--num-episodes", required=False, help="max number of episodes", default=1, type=int)
    parser.add_argument(
        "-r", required=False, help="if you want to start from a random position", default=False, action="store_true"
    )

    args = parser.parse_args()

    models_folder = "models/"
    possible_names = [
        item for item in os.listdir(models_folder)
        if os.path.isdir(os.path.join(models_folder, item)) and not item.startswith(".")
    ]

    if args.model_name not in possible_names:
        print("No agent with that name.")
        exit(1)

    main(
        args.model_name,
        args.weights,
        args.r,
        args.num_episodes,
        args.max_steps
    )
