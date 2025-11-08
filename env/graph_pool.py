import os
import torch
import numpy as np
import igraph as ig
from scipy.integrate import simpson



class Logger:
    def __init__(self, g):
        self.name = g['name']
        self.n_init = g['n_init']
        self.gcc_eps = g['gcc_eps']
        self.removals = g['removals']
        self.g_init = g['init']
        self.auc = simpson(self.gcc_eps, dx=1)



class GraphPool:
    def __init__(self, 
            graph_data: list[ig.Graph], 
            size: int, 
            rng: np.random.Generator, 
            is_val: bool = False, 
            render: bool = False,
        ):
        self.graph_data = graph_data
        self.size = size
        self.rng = rng
        self.is_val = is_val
        self.render = render

        self.graphs:  list[ig.Graph] = []

        n_slots = min(self.size, len(self.graph_data))
        for i in range(n_slots):
            g = self.graph_data[i] if self.is_val else self.rng.choice(self.graph_data)
            g = g.copy()

            n_init = g.vcount()
            g.vs["i_init"] = list(range(n_init))
            g['n_init'] = n_init; g['gcc_eps'] = []; g['removals'] = []
            g['init'] = g.copy() if self.render == 'plot' else None
            if self.is_val:
                print(f"Loaded graph {g['name']}")
            self.graphs.append(g)

        self.next_path_id = n_slots if self.is_val else None

        
    def load_new_graph(self, i: int) -> bool:
        if self.is_val:
            if self.next_path_id >= len(self.graph_data):
                self.graphs.pop(i)
                return False
            g_id = self.next_path_id
            self.next_path_id += 1
        else:
            g_id = self.rng.integers(len(self.graph_data))
            
        g = self.graph_data[g_id]
        g = g.copy()
        n_init = g.vcount()
        g.vs["i_init"] = list(range(n_init))
        g['n_init'] = n_init; g['gcc_eps'] = []; g['removals'] = []
        g['init'] = g.copy() if self.render == 'plot' else None
        if self.is_val:
            print(f"Loaded graph {g['name']}")
        self.graphs[i] = g
        return True


    def delete_nodes(self, node_array: np.ndarray):
        assert node_array.shape == (len(self.graphs),), \
            f'incorrect size for node array ({node_array.shape})'
        for i, rid in enumerate(node_array):
            original_id = self.graphs[i].vs[rid]["i_init"]
            self.graphs[i]['removals'].append(original_id)
            self.graphs[i].delete_vertices(rid)


    def prune_scc(self):
        for i, g in enumerate(self.graphs):
            cc = g.connected_components()
            cc_sizes = np.array(cc.sizes()); membership = np.array(cc.membership)
            gcc_idx = int(cc_sizes.argmax())
            threshold = 0.1 * self.graphs[i]['n_init']
            delete_mask = (cc_sizes[membership] < threshold) & (membership != gcc_idx)
            delete_idx = np.flatnonzero(delete_mask)
            if delete_idx.size:
                self.graphs[i].delete_vertices(delete_idx)


    def get_lcc_sizes(self)->tuple[np.ndarray, np.ndarray, dict[Logger]]:
        b_size = len(self.graphs)
        lcc_arr = np.empty(b_size, dtype=np.float32); done_arr = np.empty(b_size, dtype=bool)
        for i, g in enumerate(self.graphs):
            lcc_size = g.connected_components().giant().vcount() / self.graphs[i]['n_init']
            lcc_arr[i] = lcc_size; done_arr[i] = (lcc_size < 0.1)
            self.graphs[i]['gcc_eps'].append(lcc_size)
        return lcc_arr, done_arr, [Logger(self.graphs[i]) for i in np.where(done_arr)[0]]
   

    def sample_nodes(self)->np.ndarray:
        return self.rng.integers(0, [g.vcount() for g in self.graphs])
    


