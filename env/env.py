import os
import numpy as np
import igraph as ig
import pickle as pkl

from env.graph_pool import GraphPool
from utils import plot_process, load_g

class DismantleEnv:
    def __init__(self,
            data_dir: str | None = None,
            graph_data: list[ig.Graph] | None = None,
            batch_size: int=1,
            is_val: bool=False,
            seed: int=0,
            remove_scc: bool=True,
            render: bool | str = False
        ):
        if bool(data_dir) ^ bool(graph_data):
            if data_dir:
                graph_data = [load_g(os.path.join(data_dir, p), f'{os.path.basename(data_dir)}_{os.path.splitext(os.path.basename(p))[0]}') 
                              for p in sorted(os.listdir(data_dir))]
        else:
            raise ValueError("Need either data_dir or g_list (but not both).")

        self.graph_data = graph_data
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.is_val = is_val
        self.remove_scc = remove_scc
        self.render = render

    def reset(self):
        self.pool = GraphPool(
            self.graph_data, 
            self.batch_size, 
            self.rng, 
            self.is_val, 
            self.render
        )

        if self.is_val and self.remove_scc:
            self.pool.prune_scc()
        
        return [g.copy() for g in self.pool.graphs], {}

    def sample_act(self)->np.ndarray:
        return self.pool.sample_nodes()

    def step(self, act_arr):
        self.pool.delete_nodes(act_arr)

        if self.is_val and self.remove_scc:
            self.pool.prune_scc()

        lcc_arr, done_arr, info = self.pool.get_lcc_sizes()
        reward_arr = -lcc_arr

        if self.render:
            for logger in info:
                print(f'{logger.name}, AUC={logger.auc:.2f}')
                if self.render == 'plot':
                    plot_process(logger.g_init, logger.removals, logger.gcc_eps)
        return [g.copy() for g in self.pool.graphs], reward_arr, done_arr, info
    
    def reset_async(self, done_arr: np.ndarray) -> tuple[list[ig.Graph], list]:
        replaced_ids = []
        # iterate in reverse so index shifts don't break subsequent pops
        for done_id in np.flatnonzero(done_arr)[::-1]:
            replaced = self.pool.load_new_graph(done_id)
            if replaced:
                replaced_ids.append(done_id)
        return [g.copy() for g in self.pool.graphs], replaced_ids