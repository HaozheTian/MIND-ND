import torch
import numpy as np

from .graph_data import Batch
from .graph_data import ig_to_data



class ReplayBuffer():
    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        
        self.ptr = 0
        self.full = False
        self.obs_buffer = np.empty(buffer_size, dtype=object)
        self.act_buffer = np.empty(buffer_size, dtype=np.int64)
        self.obs_next_buffer = np.empty(buffer_size, dtype=object)
        self.rew_buffer = np.empty(buffer_size, dtype=np.float32)
        self.done_buffer = np.empty(buffer_size, dtype=bool)
        
    def add(self, obs_list, act_arr, obs_next_list, rew_arr, done_arr):
        num_t = len(obs_list)
        idx = np.arange(self.ptr, self.ptr+num_t)%self.buffer_size
        self.obs_buffer[idx] = np.array([ig_to_data(g) for g in obs_list])
        self.act_buffer[idx] = act_arr
        self.obs_next_buffer[idx] = np.array([ig_to_data(g) for g in obs_next_list])
        self.rew_buffer[idx] = rew_arr
        self.done_buffer[idx] = done_arr

        end = self.ptr + num_t
        if end >= self.buffer_size:
            self.full = True
        self.ptr = end % self.buffer_size
            
    def sample(self, batch_size):
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.ptr) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.ptr, size=batch_size)
            
        obs = Batch(self.device, self.obs_buffer[batch_inds].tolist())
        obs_next = Batch(self.device, self.obs_next_buffer[batch_inds].tolist())
        act = torch.tensor(self.act_buffer[batch_inds], device=self.device, dtype=torch.long)
        rew = torch.tensor(self.rew_buffer[batch_inds], device=self.device, dtype=torch.float32)
        done = torch.tensor(self.done_buffer[batch_inds], device=self.device, dtype=torch.float32)
        
        return obs, act, obs_next, rew, done