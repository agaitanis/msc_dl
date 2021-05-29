import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gym
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
from rl.policy import GreedyQPolicy
from rl.policy import BoltzmannQPolicy
from rl.policy import MaxBoltzmannQPolicy
from rl.callbacks import FileLogger
import json
import matplotlib.pyplot as plt
from gym.envs.classic_control.mountain_car import MountainCarEnv
from time import time


class MountainCarWithNoiseEnv(MountainCarEnv):
    def __init__(self, goal_velocity=0, noise_std=0):
        super().__init__(goal_velocity)
        self.noise_std = noise_std

    def step(self, action):
        state, reward, done, info = super().step(action)
        
        if self.noise_std > 0:
            state[0] += np.random.normal(0, self.noise_std)
            state[1] += np.random.normal(0, self.noise_std)
        
        return state, reward, done, info


def get_model(env):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(env.action_space.n))
    
    return model


def get_results_file_name(experiment):
    return experiment + '.json'


def exponential_smoothing(x, alpha):
    y = np.zeros_like(x)
    
    y[0] = x[0]
    
    for i in range(1, len(x)):
        y[i] = alpha*x[i] + (1-alpha)*y[i-1]
    
    return y


def plot_results(experiments, prefix):
    plt.figure()
    plt.title('Mean reward per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    for experiment in experiments:
        results_file_name = get_results_file_name(experiment)     
        with open(results_file_name, 'r') as f:
            results = json.load(f)
        episodes = results['episode']
        rewards = results['episode_reward']
        # rewards = exponential_smoothing(rewards, 0.01) # FIXME
        plt.plot(episodes, rewards, label=experiment)
    plt.legend()
    plt.show()
    plt.savefig(prefix + '_rewards.jpg')
    
    plt.figure()
    plt.title('Mean Q per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Q')
    for experiment in experiments:
        results_file_name = get_results_file_name(experiment)     
        with open(results_file_name, 'r') as f:
            results = json.load(f)
        episodes = results['episode']
        q = results['mean_q']
        plt.plot(episodes, q, label=experiment)
    plt.legend()
    plt.show()
    plt.savefig(prefix + '_q.jpg')


def run_experiment(experiment, policy, noise_std):
    print(f'\n\n------ Running experiment: {experiment} ------\n')
    
    results_file_name = get_results_file_name(experiment)
    
    try:
        env = MountainCarWithNoiseEnv(noise_std=noise_std)
        
        env.seed(0)
        np.random.seed(0)
        
        model = get_model(env)
        
        agent = DQNAgent(model=model,
                         nb_actions=env.action_space.n, 
                         memory=SequentialMemory(limit=50000, window_length=1), 
                         nb_steps_warmup=50, 
                         target_model_update=1e-2, 
                         policy=policy)
        
        callbacks=[FileLogger(results_file_name)]
        
        agent.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])
        agent.fit(env, nb_steps=150000, nb_max_episode_steps=200, callbacks=callbacks,
                  visualize=True, verbose=1) # FIXME
        
        # input("Press enter to start to testing...")
        # agent.test(env, nb_episodes=5, visualize=True)
    finally:
        env.close()


def main():
    t0 = time()
    
    print(f'tensorflow version = {tf.__version__}')
    print(tf.config.list_physical_devices('GPU'))
    print()
    
    all_params = [
        ('Greedy', GreedyQPolicy(), 0),
        ('EpsGreedy eps=0.05', EpsGreedyQPolicy(eps=0.05), 0),
        ('EpsGreedy eps=0.1', EpsGreedyQPolicy(eps=0.1), 0),
        ('EpsGreedy eps=0.2', EpsGreedyQPolicy(eps=0.2), 0),
        ('MaxBoltzmann eps=0.05', MaxBoltzmannQPolicy(eps=0.05), 0),
        ('MaxBoltzmann eps=0.1', MaxBoltzmannQPolicy(eps=0.1), 0),
        ('MaxBoltzmann eps=0.2', MaxBoltzmannQPolicy(eps=0.2), 0),
        ]
    for params in all_params:
        run_experiment(*params)
    
    experiments = [x[0] for x in all_params]
    plot_results(experiments, 'all_policies_without_noise')
    
    print(f'Total time: {(time() - t0)/60:.1f} min')


if __name__ == '__main__':
    main()