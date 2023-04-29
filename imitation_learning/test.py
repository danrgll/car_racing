from __future__ import print_function

import sys

import utils

sys.path.append("../")

from datetime import datetime
import numpy as np
import gym
import os
import json
import torch
from agent.bc_agent import BCAgent
from utils import *


def run_episode(env, agent, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0

    state = env.reset()
    
    # fix bug of curropted states without rendering in racingcar gym environment
    env.viewer.window.dispatch_events() 
    state_history = []
    while True:
        
        # preprocess the state in the same way than in your preprocessing in train_agent.py
        state = utils.rgb2gray(state)
        # predict next action
        if len(state_history) == history_length:
            # print("new action")
            state_history.pop(0)
            state_history.append(state)
            state = np.array(state_history).astype("float32")
            # print(state.shape)
            # state = state.reshape((1, history_length, 96, 96))
            # X_train = np.array(X_seq_train).astype("float32")
            # print("test")
            # print(state.shape)
            state = torch.from_numpy(state)
            state = state.to(device)
            state = state.view((1, history_length, 96, 96))
            action = agent.predict(state).detach().numpy()
            # print(action)
            # transform action back to continuous
            action = utils.id_to_action(action, max_speed=0.1)
        else:
            state_history.append(state)
            action = np.array([0])
            action = utils.id_to_action(action, max_speed=0.1)


        
        # TODO: get the action from your agent! You need to transform the discretized actions to continuous
        # actions.
        # hints:
        #       - the action array fed into env.step() needs to have a shape like np.array([0.0, 0.0, 0.0])
        #       - just in case your agent misses the first turn because it is too fast: you are allowed to clip the acceleration in test_agent.py
        #       - you can use the softmax output to calculate the amount of lateral acceleration

        next_state, r, done, info = env.step(action)
        episode_reward += r       
        state = next_state
        step += 1
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    # important: don't set rendering to False for evaluation (you may get corrupted state images from gym)
    rendering = True                      
    
    n_test_episodes = 15                  # number of episodes to test

    history_length = 3

    # TODO: load agent
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    agent = BCAgent(history_length, device, lr=5*10e-4)
    agent.load("models/agent.pt")

    env = gym.make('CarRacing-v0').unwrapped

    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=rendering)
        episode_rewards.append(episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    fname = "results/results_bc_agent-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')
