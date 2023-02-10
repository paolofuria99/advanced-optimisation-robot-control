import argparse
import os

import plot
from dqn.algorithm import DQL


def main(
        name: str,
        weights: str,
        random_start: bool,
        num_episodes: int,
        episode_duration: float,
        show_plots: bool = True
) -> None:
    network, env = DQL.load(name, weights_name=weights)

    # Compute max steps from episode duration
    max_steps = episode_duration / env.agent.sim_time_step

    print("Simulating...")
    states, torques, costs = env.render_greedy_policy(
        network,
        random_start=random_start,
        max_steps=max_steps,
        num_episodes=num_episodes,
        display=True
    )

    print("Done.")

    # Plot last episode
    if num_episodes > 0 and show_plots:
        print("Plotting...")
        idx = num_episodes - 1
        plt = plot.Plot(states[idx], torques[idx], costs[idx], network, env)
        plt.plot_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name", help="the name of the agent to evaluate", type=str)
    parser.add_argument("--weights", required=False, help="the name of the weights to load", type=str)
    parser.add_argument("--num-episodes", required=False, help="max number of episodes", default=1, type=int)
    parser.add_argument("--duration", required=False, help="duration of episode (in seconds)", default=7.0, type=float)
    parser.add_argument(
        "-r", required=False, help="if you want to start from a random position", default=False, action="store_true"
    )
    parser.add_argument(
        "-np", required=False, help="if you want to disable plotting of the last episode", default=False,
        action="store_true"
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
        args.duration,
        show_plots=not args.np
    )
