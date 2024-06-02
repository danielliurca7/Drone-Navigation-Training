# Challenges

## The set-up
The first challenge was the set-up, because I had problems with the gymnasium and stable-baselines3 packages dependencies, but I managed to find a way to make the environment work.

## The environment
During the writing of the environment, I did not encounter any problems, especially since I found an implmentation of the render function.

I chose the observation space as a vector of that contains the agent position and the target position, because this is sufficient for the agent to learn.

Note that I do not allow the agent, since it models a drone to crash into walls, in which case the episode is terminated with a big penalty.

## The training
During the training I encountered some problems, having initially as a reward function the negative of the Manhattan distance between the agent and the target.

This produced a behaviour of the agent, when starting really close to the target, of just moving one or two squares away from the circle, before finally reaching it, because, compared with the situation where the they start apart, it still receives a good reward.

In order to address this I defined a movement cost which is inverse proportional with the initial distance, which ends up being multiplyed with the percentage of episodes(current episode over the maximum number of episodes), and then multiplyed with the Manhattan distance.

This attempts to give a similar movement reward for an agent that is far away as one that is close, and to punish the agents more for spending a lot of time, which mainly remedies the unexpected behaviour of moving back and forth close to the target.


# Improvements and Extensions
Possible improvements would be to see why the DQN algorithm performs this poorly, as can be seen in the results. I tried to play with some hyperparameter, but I did not manage to get them right. It might be related to the exploration rate, since it seems to get stack in a local minimum. I can see, tho, that PPO does work well on all environment sizes, up to 100.

Of course an extension would be, working with a continous environment instead of a discrete one, but I think it just make the physics of the world and the agent control (needs to have speed and steering) more complicated, which is beyond the scope of the project.

Another interesting one will be to introduce some obstacles, that like the wall, the agent cannot run into, but since we will have to include their position in the observation, this environment will take longer to train and test and I don't have that much time left.


Documentation
>https://gymnasium.farama.org/index.html
>https://stable-baselines3.readthedocs.io/en/master/
>https://www.youtube.com/playlist?list=PLQVvvaa0QuDf0O2DWwLZBfJeYY-JOeZB1
