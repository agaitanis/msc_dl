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


def plot_results(experiments, prefix):
    for experiment in experiments:
        results_file_name = get_results_file_name(experiment)
        
        with open(results_file_name, 'r') as f:
            results = json.load(f)
            
        episodes = results['episode']
        rewards = results['episode_reward']
        mean_q = results['mean_q']
    
        plt.figure()
        plt.plot(episodes, rewards)
        plt.title('Mean reward per episode')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.show()
        plt.savefig(prefix + '_rewards.jpg')
        
        plt.figure()
        plt.plot(episodes, mean_q)
        plt.title('Mean Q per episode')
        plt.xlabel('Episodes')
        plt.ylabel('Q')
        plt.show()
        plt.savefig(prefix + '_q.jpg')


def run_experiment(experiment, policy):
    results_file_name = get_results_file_name(experiment)
    
    try:
        env = gym.make('MountainCar-v0')
        
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
        agent.fit(env, nb_steps=1500, callbacks=callbacks, visualize=True, verbose=0) # FIXME
        
        # input("Press enter to start to testing...")
        # agent.test(env, nb_episodes=5, visualize=True)
    finally:
        env.close()


def main():
    print(f'tensorflow version = {tf.__version__}')
    print(tf.config.list_physical_devices('GPU'))
    print()
    
    all_params = [
        ("EpsGreedy", EpsGreedyQPolicy()),
        ("Greedy", GreedyQPolicy()),
        ("Boltzmann", BoltzmannQPolicy()),
        ("MaxBoltzmann", MaxBoltzmannQPolicy()),
        ]
    
    experiments = [x[0] for x in all_params]
    
    for params in all_params:
        run_experiment(*params)
    
    plot_results(experiments, 'all_policies_without_noise')


if __name__ == '__main__':
    main()