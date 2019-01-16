from unityagents import UnityEnvironment
import numpy as np
env = UnityEnvironment(file_name='Reacher_Windows_x86_64/Reacher.exe')



fc1_units = 512  #128 ## 64
fc2_units = 256  #64 #32
fc3_units = 64
#fc4 is 16 by default
lr_actor = 1e-3
lr_critic = 1e-3
buffer_size = int(1e6)
batch_size = 64 # 128
Gamma = 0.999
tau = 0.001

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np




def layer_init(self):
    in_w = self.weight.data.size()[0]
    lim = 1./(np.sqrt(in_w))
    return (-lim,lim)


class Actor(nn.Module):
    ## Module is base class for all neural networks
    
    """ Define the class for Actor netorks to be used in the model
    
    Parameters:
    state_size, action_size, seed,fc1_units, fc2_units 
    all int.
    
    """
    def __init__(self, state_size, action_size, seed,fc1_units = fc1_units , fc2_units = fc2_units, fc3_units = fc3_units):        
        super(Actor,self).__init__()
        ## just writing super() also enough    
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units,fc3_units)
        self.fc4 = nn.Linear(fc3_units,16)
        self.fc5 = nn.Linear(16,action_size)
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()
        
    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x =torch.tanh(self.fc5(x))
        return x
    
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*layer_init(self.fc3))
        self.fc4.weight.data.uniform_(*layer_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3,3e-3)
        ### also try just using the reset_parameters()method of layers      
    
    
class Critic(nn.Module):
    """
    Define the class for Critic netorks to be used in the model
    
    Parameters:
    state_size, action_size, seed,fc1_units, fc2_units 
    all int.
    
    """
    
    def __init__(self, state_size, action_size, seed,fc1_units = fc1_units, fc2_units = fc2_units , fc3_units = fc3_units):
        super(Critic,self).__init__()
        ## just writing super() also enough  
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units,fc3_units)
        self.fc4 = nn.Linear(fc3_units,16)
        self.fc5 = nn.Linear(16,1)
        
        self.seed = torch.manual_seed(seed)
        self.reset_parameters()
        
    def forward(self, state, action):
        
        xst = F.relu(self.fc1(state))
        x =torch.cat((xst, action), dim = 1) ## concatenate along dimension 1 (columns)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def reset_parameters(self):
        
        self.fc1.weight.data.uniform_(*layer_init(self.fc1))
        self.fc2.weight.data.uniform_(*layer_init(self.fc2))
        self.fc3.weight.data.uniform_(*layer_init(self.fc3))
        self.fc4.weight.data.uniform_(*layer_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3,3e-3)
        ### also try just using the reset_parameters()method of layers      
    
    
        