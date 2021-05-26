import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import gym
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import MaxBoltzmannQPolicy
from rl.callbacks import FileLogger
import json
import matplotlib.pyplot as plt

print(f'tensorflow version = {tf.__version__}')
print(tf.config.list_physical_devices('GPU'))
print()

log_name = 'log.json'

try:
    env = gym.make('MountainCar-v0')
    
    env.seed(0)
    np.random.seed(0)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dense(env.action_space.n))
    
    agent = DQNAgent(model=model,
                     nb_actions=env.action_space.n, 
                     memory=SequentialMemory(limit=50000, window_length=1), 
                     nb_steps_warmup=50, 
                     target_model_update=1e-2, 
                     policy=MaxBoltzmannQPolicy())
    
    callbacks=[FileLogger(log_name)]
    
    agent.compile(keras.optimizers.Adam(lr=1e-3), metrics=['mae'])
    agent.fit(env, nb_steps=150, callbacks=callbacks, visualize=True, verbose=0) # FIXME
    
    # input("Press enter to start to testing...")
    # agent.test(env, nb_episodes=5, visualize=True)
finally:
    env.close()
    
with open(log_name, 'r') as f:
    data = json.load(f)
    
episodes = data['episode']
rewards = data['episode_reward']
mean_q = data['mean_q']

plt.figure()
plt.plot(episodes, rewards)
plt.title('Mean reward per episode')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.show()

plt.figure()
plt.plot(episodes, mean_q)
plt.title('Mean Q per episode')
plt.xlabel('Episodes')
plt.ylabel('Q')
plt.show()