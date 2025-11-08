import torch
import numpy as np

def ig_to_data(g):
    g = g.copy(); n = g.vcount()
    g.to_directed(); g.add_vertices(1); g.add_edges([(v_id, n) for v_id in range(n)])
    return Graph(np.array(g.get_edgelist(), dtype=np.int64).T, n)

class Graph:
    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index  # shape (2, E)
        self.num_nodes = num_nodes

class Batch:
    def __init__(self, device, graph_array):
        self.device = device
        self.batch_size = len(graph_array)

        num_nodes_b = np.array([g.num_nodes+1 for g in graph_array], dtype=np.int64)
        start_ids = np.zeros(self.batch_size, dtype=np.int64)
        start_ids[1:] = np.cumsum(num_nodes_b[:-1]) # [0, N1, N2, ...]

        act_offsets = start_ids - np.arange(self.batch_size)

        omni_ids = start_ids + num_nodes_b - 1

        batch = np.repeat(np.arange(self.batch_size), num_nodes_b, axis=0)

        edge_index = np.concatenate([g.edge_index+s for s, g in zip(start_ids, graph_array)], axis=1)

        self.num_nodes_b = torch.tensor(num_nodes_b, device=self.device)
        self.total_nodes = self.num_nodes_b.sum()

        self.act_offsets = torch.tensor(act_offsets, device=self.device)

        self.omni_ids = torch.tensor(omni_ids, device=self.device)
        self.non_omni_mask = torch.ones(self.total_nodes, dtype=torch.bool, device=self.device)
        self.non_omni_mask[self.omni_ids] = False

        self.edge_index = torch.tensor(edge_index, device=self.device)
        self.batch = torch.tensor(batch, device=self.device)
        self.batch_non_omni = self.batch[self.non_omni_mask]