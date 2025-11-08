
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm

from utils.graph_data import Batch



class MINDConv(nn.Module):
    def __init__(self, F, H):
        super().__init__()
        assert F%H == 0, f'num_features {F} not divisible by num_heads {H}'
        self.F, self.H = F, H

        self.W_src = nn.Linear(F, F)
        self.W_dst = nn.Linear(F, F)
        self.mlp_a_src = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(F, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, H)
        )
        self.mlp_a_dst = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(F, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, H)
        )

    def forward(self, h, edge_index):
        src, dst = edge_index  # message passing direction: src -> dst
        N, F, H = h.shape[0], self.F, self.H
        D, E = F//H, len(src)

        g_src = self.W_src(h) # -> (N, F)
        g_dst = self.W_dst(h) # -> (N, F)

        msg_src, msg_dst = g_src[src], g_dst[dst] # -> (E, F)
        
        a_src = torch.sigmoid(self.mlp_a_src(msg_src+msg_dst)).unsqueeze(-1) # -> (E, H, 1)
        a_dst = torch.sigmoid(self.mlp_a_dst(g_dst)).unsqueeze(-1)           # -> (N, H, 1)
        
        # Get message from source nodes -> (N, H, D)
        h_next = a_dst * g_dst.view(N, H, D) \
                 + scatter_add(a_src * msg_src.view(E, H, D), dst, 0, dim_size=N)
        return h_next.view(N, -1)



class MIND(nn.Module):
    def __init__(self, num_features, num_heads, num_mps):
        super().__init__()
        self.num_features = num_features
        self.register_buffer("x_init", torch.ones(1, num_features))
        self.num_mps = num_mps
        self.convs = nn.ModuleList([MINDConv(num_features, num_heads) for _ in range(num_mps)])
        self.graph_norm = GraphNorm(num_features*num_mps, eps=1e-4)

    def forward(self, g: Batch):
        x_profile = torch.empty(g.total_nodes, self.num_features*self.num_mps, device=self.x_init.device)
        x_k = self.x_init.expand(g.total_nodes, -1)
        for k, conv in enumerate(self.convs):
            x_k = conv(x_k, g.edge_index)
            x_profile[:, k*self.num_features : (k+1)*self.num_features] = x_k
            x_k = torch.relu(x_k)
        
        x_profile = self.graph_norm(x_profile, g.batch)
        x_profile = torch.cat([
            x_profile[g.non_omni_mask],
            x_profile[g.omni_ids][g.batch_non_omni]
        ], dim=1)
        return x_profile
