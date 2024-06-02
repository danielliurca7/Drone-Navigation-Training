import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


# The environment class
class DroneNavigation(gym.Env):

    def __init__(self, size: int = 10, max_steps: int = None, agent_location: np.array = None, target_location: np.array = None) -> None:
        self.size = size         # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.episode_step = 0    # The step number in the current iteration

        # The maximum number of steps the agent can take
        # If it is not given, make it 5 times the size of the environment
        # This should give the agent pleanty of time to explore
        if max_steps is None:
            self._max_steps = 5 * self.size
        else:
            self._max_steps = max_steps

        # Make sure there is no seed set
        np.random.seed(None)

        # Initialize the agent's starting location
        if agent_location is not None:
            self._agent_location = agent_location
        else:
            self._agent_location = np.random.randint(self.size, size=2)

        # Initialize the target's starting location
        if target_location is not None:
            self._target_location = target_location
        else:
            self._target_location = np.random.randint(self.size, size=2)

        # Initialize the cost of making a _movement_cost
        # It depends on the agent starting location, because if it starts close to target it is hard
        # to learn to not waste time until it get to the target, because it is not punished enough.
        self._movement_cost = 1 + 1 / np.linalg.norm(self._agent_location - self._target_location, ord=1)

        # Observations are a vector with the agent's and the target's location concatenated.
        # All agent and location position have the size of the environment
        self.observation_space = spaces.Box(0, size - 1, shape=(4,), dtype=int)

        # We have 4 actions, corresponding to "right", "down", "left", "up"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "down" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # down
            2: np.array([-1, 0]), # left
            3: np.array([0, -1]), # up
        }

        self.window = None
        self.clock = None


    # The function that return the observation, which includes the agent and the target location
    def _get_obs(self) -> np.array:
        return np.concatenate((self._agent_location, self._target_location))


    # The function that return the information about the environment
    def _get_info(self) -> dict:
        return {"agent": self._agent_location, "target_location": self._target_location, "distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}


    # The function that return whether the agent reached his target
    def _get_terminated(self) -> bool:
        return np.array_equal(self._agent_location, self._target_location)


    # The function that tells us whether or not the drone collided with a wall_collision
    # We will need it, because we forbid collision with walls
    def _wall_collision(self, prev_agent_location):
        return np.array_equal(self._agent_location, prev_agent_location)


    # The function that tells us whether the episode ended, either because the agent hit a wall, or because the maximum steps were reached
    def _get_truncated(self, prev_agent_location) -> bool:
        wall_collision = self._wall_collision(prev_agent_location)

        return self.episode_step == self._max_steps or wall_collision, wall_collision


    # The reward fuction return a large reward in case of reaching the target, and a negative one in case of wall collision
    # The reward for a movement will be explained in the documentation, but depends on the initial and the current distance between the target and the current step
    def _reward(self, terminated: bool, wall_collision: bool) -> float:
        if terminated:
            return self.size * self.size

        if wall_collision:
            return - self.size * self.size

        return - (self.episode_step / self._max_steps * self._movement_cost) * np.linalg.norm(self._agent_location - self._target_location, ord=1)


    # The reset function will reset all the environment parameters
    def reset(self, seed=None, options=None) -> tuple[np.array, dict]:
        # Reset the step number
        self.episode_step = 0

        # Set the seed if needed
        if seed is not None:
            np.random.seed(seed)

        # Choose the agent's location uniformly at random
        self._agent_location = np.random.randint(self.size, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.random.randint(self.size, size=2)

        # Set the movement cost
        self._movement_cost = 1 + 1 / np.linalg.norm(self._agent_location - self._target_location, ord=1)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info


    # The step function returns the information for a step, given an action
    def step(self, action: int) -> tuple[np.array, float, bool, bool, dict]:
        # Increase the step number
        self.episode_step += 1

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Save the agent location before stepping
        prev_agent_location = np.copy(self._agent_location)

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Get all the step information from the coresponding private fuctions
        observation = self._get_obs()
        info = self._get_info()

        terminated                 = self._get_terminated()
        truncated, wall_collision  = self._get_truncated(prev_agent_location)

        reward = self._reward(terminated, wall_collision)

        return observation, reward, terminated, truncated, info


    # The funtion that renders the environment, the target a red square and the drone a blue circle
    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )


        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(3)


    # Closes the window necessary for rendering
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
