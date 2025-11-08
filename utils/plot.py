import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_degree_distribution(g):
    deg = g.degree()
    values, counts = np.unique(deg, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.loglog(values, counts/len(deg), "o-", lw=1.5)
    plt.xlabel("Degree (k)")
    plt.ylabel("P(k)")
    plt.title("Degree distribution (log-log)")
    plt.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_graph(
    graph, 
    ax=None,
    c="#777777",
    layout=None, 
    removals=None, 
    node_size=2
    ):
    if ax is None:
        _, ax = plt.subplots()
    layout = graph.layout("fr") if layout == None else layout
    
    edge_array = np.array(graph.get_edgelist())

    node_colors = np.full(graph.vcount(), c, dtype=object)        
    edge_colors = np.full(len(edge_array), "#777777", dtype=object)
    if removals:
        node_colors[removals] = "#e56b6f"
        edge_mask = np.isin(edge_array[:, 0], removals) | np.isin(edge_array[:, 1], removals)
        edge_colors[edge_mask] = "#e56b6f"
        
    coords = np.array(layout.coords)
    
    if len(edge_array) > 0:
        ax.add_collection(
            LineCollection(
                np.stack([coords[edge_array[:, 0]], coords[edge_array[:, 1]]], axis=1), 
                colors=edge_colors, linewidths=1.0
            )
        )
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=node_colors,
        s=node_size,
        edgecolors='black',
        linewidths=0.5,
        zorder=2
    )
        
    ax.set_aspect('equal')
    ax.axis('off')
    return ax

def plot_process(init_graph, removals, gcc_list):
    layout = init_graph.layout('fr')

    N = len(removals)
    selected = np.linspace(0, N-1, num=4, dtype=int)[1:]
    
    N_selected = len(selected)
    _, axes = plt.subplots(1, N_selected+1, figsize=((N_selected+1)* 3, 3))
    for i in range(N_selected): 
        plot_graph(init_graph, axes[i], layout=layout, removals=removals[:selected[i]])
        axes[i].set_title(f'step {selected[i]}')
        
    axes[-1].plot(gcc_list, c="#e56b6f", linewidth=2)
    axes[-1].plot([0, len(gcc_list)], [gcc_list[-1], gcc_list[-1]], ':' ,c='#777777', linewidth=2)
    axes[-1].set_xlim([0, init_graph.vcount()])
    axes[-1].set_xticks([])
    axes[-1].set_yticks([])
    axes[-1].set_ylim([0, 1])
    axes[-1].set_ylabel('LCC(x)/|V|')
    axes[-1].spines['top'].set_visible(False)
    axes[-1].spines['right'].set_visible(False)
    axes[-1].spines['left'].set_linewidth(2)
    axes[-1].spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    plt.show()