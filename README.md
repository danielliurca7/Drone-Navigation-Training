# Package installation

This is a configuration on an Arch Linux environment and I did have some trouble with the configuration.
Hopefully it all works fine for you :)

```
pip3 install torch torchvision torchaudio
pip3 install swig tensorboard gymnasium stable_baselines3 Shimmy
pip3 install cloudpickle==3.0.0
```


# The main function

The main command of the project will be running the main.py file. In order to see what each parameter does you can run:

```
python main.py --help
```

For reference, I will also write them here:

--action   REQUIRED   What action to test. Values can be:
>train_and_render (trains the models and renders a playout),
>train_and_save (trains and saves the models to the ./models folders)
>load_and_render (loads renders the models and play a number of episodes)

--size REQUIRED  The environment size. Value must be an integer.

--max_steps   The environment maximum steps before truncation. Value must be an integer.

--agent_location   The agent's starting location. Value must be a list with comma separated integer ex. [1,2].

--target_location   The targets's starting location. Value must be a list with comma separated integer ex. [1,2].

--iterations   The training iterations for an algorithm. Value must be an integer.

--timesteps   The training timesteps per one iteration. Value must be an integer.

--algorithms   The training algorithms to train or test. Value must be a list with comma separated strings ex. [a2c,dqn,ppo].

--model_version   The timesteps of the model that you wanna use, default being the latest model. Value must be an integer.

--render_episodes The number of episodes to render, in case of rendering. Value must be an integer.


## Quick commands exaples

Train and visualize the results for an environment of size 5, for 3 iterations, for PPO algorithm:
```
python main.py --action train_and_render --size 5 --iterations 3 --algorithms=[ppo]
```

Train and save the results(without visualisation) for an environment of size 5, for 3 iterations, for PPO algorithm:
```
python main.py --action train_and_save --size 5 --iterations 3 --algorithms=[ppo]
```

Visualize the results using an algorithm already trained for an environment of size 5, for PPO algorithm:
```
python main.py --action load_and_render --size 5 --algorithms=[ppo]
```

An example with all parameters set:
```
python main.py --action train_and_render --size 5 --max_steps 20 --agent_location [1,2] --target_location [3,4] --iterations 1 --timesteps 1024 --algorithms [ppo] --model_version 10240 --render_episodes 5
```

Note: not every parameter is relevant for every action.

KEEP IN MIND: All this commands will create in the ./logs, ./models directory a folder for each environment size, where the experiments will be saved.

TIP: Watch my traning results, as explained below, before attempting to test training, because the data will be overwritten for a specific environment size


# Training results

In order to see the training results, the ones that I got or some that you get by training you need to open Tensorboard, by first executing a command in terminal, in the project's folder, like:

```
python3 -m tensorboard.main --logdir=logs/100
```

Note: you can visualize the training data for every environment size, by changing the number in the command, ex. --logdir=logs/10, --logdir=logs/20, etc.

Then you need to open your browser and go to the address:

```
http://localhost:6006/#scalars
```

There you will see the metrics like the average episode length or the average reward, for every training step.
If you render a model that does not do well, you can watch how the training went.

If you are in the process of training a algorithm, you can see the data updates in real time, by going to options and checking the "Reload data" box.
