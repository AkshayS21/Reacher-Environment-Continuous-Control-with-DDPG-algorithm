from unityagents import UnityEnvironment
import numpy as np

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

fc1_units = 512 #512  #128 ## 64
fc2_units = 256 #256  #64 #32
fc3_units = 64
#fc4 is 16 by default
lr_actor = 1e-4
lr_critic = 1e-4
buffer_size = int(1e6)
batch_size = 128 # 128
Gamma = 0.99
tau = 0.001


agent = Agent(n_agents = 20,state_size = 33, action_size = 4, random_seed = 2)

from collections import deque
n_agents = 20


def cont_control(n_episodes = 1000,  print_every = 100 ):
    
    scores = []
    scores_deque = deque(maxlen=100)
    
    for i_episode in range(1,n_episodes+1):
        
        total_r = np.zeros(n_agents)
        #total_rewards = 0
        #state = env.reset()
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        statesA = env_info.vector_observations
        #state = env_info.vector_observations[0]
        #for t in range(max_t):
        while True:
            
            actionsA = agent.act(statesA)
            env_info = env.step(actionsA)[brain_name]
            next_statesA = env_info.vector_observations
            #next_state, reward, done,_ = env.step(action)
            rewardsA = env_info.rewards
            donesA = env_info.local_done            
            
            agent.step(statesA, actionsA, rewardsA, next_statesA, donesA)
            statesA = next_statesA
            total_r += rewardsA
            if np.any(donesA):
                break
        
        scores.append(total_r)
        scores_deque.append(total_r)
        print('\r Episode:{} \t Average Score: {:.2f} \t Average of last 100 episodes: {:.2f}'\
              .format(i_episode, np.mean(scores), np.mean(scores_deque)), end = "")
        #print(' Episode:{} \t Average Score: {}'.format(i_episode, np.mean(scores_deque)))
        #if i_episode % print_every == 0:
            
            #print('\n Episode:{} \t Average Score: {:.2f}\n'.format(i_episode, np.mean(scores_deque)), end='')
            
        if np.mean(scores_deque) >=30:
            torch.save(agent.actor_local.state_dict(), 'actor_model.pth')
            torch.save(agent.critic_local.state_dict(), 'critic_model.pth')
            print('\n Environment Solved in {} episodes, Average score:{}'.format( (i_episode - 100), np.mean(scores_deque)), end='' )
            #torch.save(agent.actor_local.state_dict(), 'actor_model.pth')
            #torch.save(agent.critic_local.state_dict(), 'critic_model.pth')
            break
            
    
    return scores

scores = cont_control()
