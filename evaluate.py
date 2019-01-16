import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from collections import namedtuple, deque

import numpy as np

import random

from model import Actor,Critic



#change this to your Reacher.exe path
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]




agent.actor_local.load_state_dict(torch.load('actor_model.pth',map_location= 'cpu'))
agent.critic_local.load_state_dict(torch.load('critic_model.pth', map_location = 'cpu'))


for episode in range(3):
    env_info = env.reset(train_mode=False)[brain_name]        
    states = env_info.vector_observations       
    score = np.zeros(num_agents)               
    
    while True:
        actions = agent.act(states, add_noise=False)                    
        
        env_info = env.step(actions)[brain_name]        
        next_states = env_info.vector_observations     
        rewards = env_info.rewards       
        dones = env_info.local_done
        score += rewards
        states = next_states

        if np.any(dones):                              
            break

    print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score))) 
