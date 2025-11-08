import torch
import torch.nn as nn
from torch_scatter import scatter_log_softmax, scatter_max

from networks.mind import MIND
from utils.graph_data import Batch



def load_dismantler(F, H, K, device, ckpt_pth=None):
    policy = SACPolicy(F, H, K).to(device)
    qf1 = SACQNetwork(F, H, K).to(device)
    qf2 = SACQNetwork(F, H, K).to(device)
    qf1_target = SACQNetwork(F, H, K).to(device)
    qf2_target = SACQNetwork(F, H, K).to(device)
    if ckpt_pth != None:
        ckpt = torch.load(ckpt_pth, weights_only=True)
        policy.load_state_dict(ckpt['policy_state_dict'])
        qf1.load_state_dict(ckpt['qf1_state_dict'])
        qf2.load_state_dict(ckpt['qf2_state_dict'])
        qf1_target.load_state_dict(ckpt['qf1_target_state_dict'])
        qf2_target.load_state_dict(ckpt['qf2_target_state_dict'])
    else:
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

    return policy, qf1, qf2, qf1_target, qf2_target



class SACPolicy(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.graph_embedding = MIND(num_features, num_heads, num_mps)
        e_size = (num_features*num_mps)*2

        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, g:Batch):
        e = self.graph_embedding(g)
        logits = self.mlp(e).flatten()
        return logits
    
    def get_action(self, g:Batch, val=False):
        logits = self(g)
        log_probs = scatter_log_softmax(logits, g.batch_non_omni, dim_size=g.batch_size)
        if val:
            _, act = scatter_max(log_probs, g.batch_non_omni, dim_size=g.batch_size)
        else:
            # Gumbel-Max trick
            gumbel_noise = -torch.empty_like(log_probs).exponential_().log()
            gumbel_logits = log_probs + gumbel_noise
            _, act = scatter_max(gumbel_logits, g.batch_non_omni, dim_size=g.batch_size)  # (B,)
        act -= g.act_offsets
        return act, log_probs



class SACQNetwork(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.graph_embedding = MIND(num_features, num_heads, num_mps)
        e_size = (num_features*num_mps)*2

        self.mlp = nn.Sequential(
            nn.Linear(e_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, g:Batch):
        e = self.graph_embedding(g)
        q_vals = self.mlp(e).flatten()
        return q_vals