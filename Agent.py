import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import copy
from collections import namedtuple, deque

import numpy as np

import random

from model import Actor,Critic



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Agent():
    
    def __init__(self, n_agents, state_size, action_size, random_seed):
        
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        
        # Actor networks local and target
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = lr_actor)
        
        ## Critic networks, local and target
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = lr_critic)
        
        self.noise = OUNoise((n_agents,action_size), random_seed)
        
        self.memory = Replay_Buffer(action_size, buffer_size, batch_size, random_seed)
        
    ## based on the random state generated by the environment, our agent will generate 
    ## an action, by passing the state thru the actor_local network
    
    def act(self,state, add_noise = True):
        
        state = torch.from_numpy(state).float().to(device) # convert state from numpy array to a tensor
        
        self.actor_local.eval() ## put network in evaluation mode, no trainig
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy() # 
            #get action, push data to cpu and convert to numpy array, for the gym env 
            
        self.actor_local.train()
        ## put the network in trainig mode for learn part
        
        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1,1)
        return action
    
    ## The action generated by the actor_local network will be passed to the env.step() function
    ## to generate next_state, rewards, done.. The agent will then add these to the memory as experiences
    ## and if we have enough experiences(>128),  the agent will sample and learn from them
    
    def step(self, state, action, reward, next_state, done):
        
        for i in range(self.n_agents):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])
        
        #self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, Gamma)
            
    
    def learn(self, experiences, Gamma):
        
        states, actions, rewards, next_states, dones = experiences
        
        ### Get Q_target for next_states to compute Q_target for current state
        
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        Q_targets = rewards + (Gamma * Q_targets_next * (1-dones))
        
        ### Compute the actual Q_target based on current critic_local network
        
        Q_expecteds = self.critic_local(states, actions)
        
        Q_loss = F.mse_loss(Q_expecteds, Q_targets)
        
        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        ### We want to train the actor_local such that it maximizes the value Q
        
        actions_pred = self.actor_local(states)
        Q_value = - self.critic_local(states, actions_pred).mean()
        
        self.actor_optimizer.zero_grad()
        Q_value.backward()
        self.actor_optimizer.step()
        
        ### soft update target params
        
        self.soft_update( self.actor_local, self.actor_target, tau)
        self.soft_update( self.critic_local, self.critic_target, tau)       
    
    def soft_update(self, local_model, target_model, tau):
        
        for target_params,local_params in zip(target_model.parameters(), local_model.parameters()):
            target_params.data.copy_( tau * local_params.data + (1-tau) * target_params.data)
    
    def reset(self):
        self.noise.reset()
    
            
            