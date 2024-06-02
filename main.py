import sys
import argparse
import numpy as np

from environment import DroneNavigation
from model import ModelWrapper


# The main function for the
if __name__ == "__main__":
    # Define the arguments accepted from console
    parser=argparse.ArgumentParser()

    parser.add_argument("--action", choices=["load_and_render", "train_and_save", "train_and_render"], type=str, help="What action to test", required=True)

    parser.add_argument("--size", help="The environment size", type=int, required=True)
    parser.add_argument("--max_steps", help="The environment maximum steps", type=int)
    parser.add_argument("--agent_location", help="The agent's starting location", type=list[int])
    parser.add_argument("--target_location", help="The targets's starting location", type=list[int])

    parser.add_argument("--iterations", help="The training iterations", type=int)
    parser.add_argument("--timesteps", help="The training timesteps per one iteration", type=int)
    parser.add_argument("--algorithms", help="The training algorithms to train or test", type=list[str])
    parser.add_argument("--model_version", help="The timesteps of the model that you wanna use", type=int)

    parser.add_argument("--render_episodes", help="The number of episodes to render", type=int)


    args=parser.parse_args()



    # Initialize the environment based of the arguments from console
    env_args = {}

    if args.max_steps is not None:
        env_args["max_steps"] = args.max_steps
    if args.agent_location is not None:
        env_args["agent_location"] = np.array([int(agent_location) for agent_location in args.agent_location[1::2]])
    if args.target_location is not None:
        env_args["target_location"] = np.array([int(target_location) for target_location in args.target_location[1::2]])

    env = DroneNavigation(size=args.size, **env_args)
    env.reset()


    # Initialize the model wrapper based of the arguments from console
    model_args = {}

    if args.iterations is not None:
        model_args["iterations"] = args.iterations
    if args.timesteps is not None:
        model_args["timesteps"] = args.timesteps
    if args.algorithms is not None:
        model_args["algorithms"] = ''.join(args.algorithms[1:-1]).split(',')

    model_wrapper = ModelWrapper(env, **model_args)


    no_episodes = args.render_episodes if args.render_episodes is not None else 5


    # Based of the action
    # For clarification check README
    match args.action:
        case "train_and_render":
            model_wrapper.train_models(save=False)

            for algorithm in model_wrapper.algorithms:
                model_wrapper.render_results(algorithm=algorithm, episodes=no_episodes)

        case "train_and_save":
            model_wrapper.train_models(save=True)

        case "load_and_render":
            for algorithm in model_wrapper.algorithms:
                model_wrapper.load_saved_model(algorithm)
                model_wrapper.render_results(algorithm=algorithm, episodes=no_episodes)

    env.close()
