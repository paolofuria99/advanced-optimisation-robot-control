import argparse
import os

from dqn.algorithm import DQL


def main(name: str) -> None:
    network, env = DQL.load(name)
    env.render_greedy_policy(network)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", help="the name of the agent to evaluate", type=str)

    args = parser.parse_args()

    models_folder = "models/"
    possible_names = [
        item for item in os.listdir(models_folder)
        if os.path.isdir(os.path.join(models_folder, item)) and not item.startswith(".")
    ]

    if args.model_name not in possible_names:
        print("No agent with that name.")
        exit(1)

    main(args.model_name)
