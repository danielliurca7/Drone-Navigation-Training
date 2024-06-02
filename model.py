import os
import gymnasium as gym

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

from environment import DroneNavigation


#The stable_baselines3 algorithm models wrapper class
class ModelWrapper:

    def __init__(self, env: gym.Env, iterations: int = 10, timesteps: int = 10240, algorithms: list[str] = ["a2c", "dqn", "ppo"]) -> None:
        # Define the environment
        self._env = env

        # Define the directories for the log and
        self._models_dir = f"models/{self._env.size}"
        self._log_dir    = f"logs/{self._env.size}"

        # The number of train iterations
        self._iterations = iterations

        # The number of steps per one train iteration
        self._timesteps = timesteps

        self.algorithms = algorithms

        # Initialize the models we are intending to test them
        self._models = {}

        if "a2c" in algorithms:
            self._models["a2c"] = A2C("MlpPolicy", self._env, tensorboard_log=self._log_dir)
        if "dqn" in algorithms:
            self._models["dqn"] = DQN("MlpPolicy", self._env, tensorboard_log=self._log_dir)
        if "ppo" in algorithms:
            self._models["ppo"] = PPO("MlpPolicy", self._env, tensorboard_log=self._log_dir)

        self._create_dirs()


    # The function that creates the directories for saving models and logs, if they do not exist
    def _create_dirs(self):
        if not os.path.exists(self._models_dir):
            os.makedirs(self._models_dir)

        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        for name in self._models:
            model_log_dir = f"{self._models_dir}/{name}"

            if not os.path.exists(model_log_dir):
                os.makedirs(model_log_dir)


    # Auxiliary function for saving the model in the correct directory
    def _save_model(self, name: str, model: OffPolicyAlgorithm | OnPolicyAlgorithm, current_step: int) -> None:
        path = f"{self._models_dir}/{name}/{current_step}"

        model.save(path)


    # The function that trains all the models defined in the class
    # It is parametized to save and log the training process
    def train_models(self, save = True, log = True) -> None:
        for i in range(1, self._iterations + 1):
            if log: print(f"Iteration  no. {i}/{self._iterations}")

            for name, model in self._models.items():
                if log: print(f"\t{name.upper()} Algorithm")

                model.learn(total_timesteps=self._timesteps, reset_num_timesteps=False, progress_bar=log)
                if save: self._save_model(name, model, i * self._timesteps)

                if log: print()

            if log: print()
            if log: print()


    # Load a model from the models directory
    # Can specify the version of it, but the default is the latest
    def load_saved_model(self, name: str, version: int = None) -> None:
        dir = f"{self._models_dir}/{name}"

        if version is None:
            version = max([int(filename.split('.')[0]) for filename in os.listdir(dir)])

        path = f"{dir}/{version}"

        if name == "a2c":
            self._models[name] = A2C.load(path)
        elif name == "dqn":
            self._models[name] = DQN.load(path)
        elif name == "ppo":
            self._models[name] = PPO.load(path)


    # The function to render play-out episodes, for a number of episodes given a parameter
    def render_results(self, algorithm: str, episodes: int = 10) -> None:
        print(f"Episode for algorithm {algorithm.upper()}")
        print()

        for i in range(1, episodes + 1):
            total_reward = 0
            obs, _ = self._env.reset()

            while True:
                action, _ = self._models[algorithm].predict(obs)
                obs, reward, terminated, truncated, info = self._env.step(action.item())

                total_reward += reward

                self._env.render()

                if terminated or truncated:
                    print(f"Finished episode {i} with reward {total_reward} " + ("and reached the target" if terminated else "and did not reach the target"))
                    break

        print()
        print()
