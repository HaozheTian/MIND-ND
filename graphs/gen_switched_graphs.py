import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import pickle


def LPA(N, m, gamma):
    g = ig.Graph(n=m+1)
    for nidx in range(g.vcount()):
        g.add_edges([(nidx, i) for i in range(nidx+1, g.vcount()) if i != nidx])
    a = m * (gamma - 3)
    for nidx in range(N - m - 1):
        node_count = g.vcount()
        node_weights = [g.degree(i) + a for i in range(node_count)]
        if np.sum(node_weights) == 0:
            node_weights = np.ones(node_count, dtype=float)
        node_weights /= np.sum(node_weights)
        end_nodes = np.random.choice(np.arange(node_count), m, p=node_weights, replace=False)
        g.add_vertex()
        g.add_edges([(node_count, i) for i in end_nodes])
    return g


def copying_model(N, m, gamma):
    g = ig.Graph(n=m+1)
    for nidx in range(g.vcount()):
        g.add_edges([(nidx, i) for i in range(nidx+1, g.vcount()) if i != nidx])
    alpha = (2 - gamma) / (1 - gamma)
    if not 0 < alpha < 1:
        raise Exception("Alpha needs to be between 0 and 1")
    for nidx in range(m + 1, N):
        g.add_vertex()
        for stub in range(m):
            if np.random.rand() < alpha:
                while True:
                    rand_endpoint = np.random.randint(nidx)
                    if not g.are_adjacent(nidx, rand_endpoint):
                        g.add_edge(nidx, rand_endpoint)
                        break
            else:
                while True:
                    rand_node = np.random.randint(nidx)
                    rand_endpoint = np.random.choice(g.neighbors(rand_node))
                    if not g.are_adjacent(nidx, rand_endpoint):
                        g.add_edge(nidx, rand_endpoint)
                        break
    return g


def switch(g, order, type):
    swt_trials = 0
    while True:
        swt_trials += 1
        if swt_trials > 100:
            return False
        e1, e2 = np.array(g.get_edgelist())[np.random.choice(g.ecount(), 2, replace=False)]
        rand_idx = np.random.randint(2, size=2)
        i, l = e1[rand_idx[0]], e1[1 - rand_idx[0]]
        j, k = e2[rand_idx[1]], e2[1 - rand_idx[1]]
        if g.are_adjacent(i, k) or g.are_adjacent(j, l) or len(list({i, j, k, l})) < 4:
            continue
        if (order[i] - order[j]) * (order[k] - order[l]) * type >= 0:
            g.delete_edges([tuple(e1), tuple(e2)])
            g.add_edges([(i, k), (j, l)])
            break
    return g


net_dict = {}
for net_no in range(10000):
    topology = np.random.choice(['LPA', 'Copy', 'ER'])
    N = 100 + np.random.randint(101)
    gamma = 2.5 + np.random.rand()
    m = np.random.choice([1, 2, 3, 4, 5, 6, 8, 10],
                         p=[1/12, 2/12, 2/12, 2/12, 2/12, 1/12, 1/12, 1/12])
    switch_type = np.random.randint(3) - 1
    r_coeff = 0.05 if switch_type == 0 else switch_type * np.random.choice([0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

    print(net_no, ':', topology, ', N =', N, ', m =', m,
          'gamma = ', gamma, ', r_target =', r_coeff, switch_type)

    trial = 0
    while True:
        if trial > 100:
            print('Network regeneration trials maxed out, increasing m to', m + 1)
            trial = 0
            m += 1

        match topology:
            case 'ER':
                net = ig.Graph.Erdos_Renyi(n=N, p=((N - 1) * m - 1) / (N * (N - 1)))
            case 'Copy':
                net = copying_model(N, m, gamma)
                if m == 1:
                    r_coeff = 0.01 + 0.04 * np.random.rand() if switch_type == 0 else \
                              switch_type * np.random.choice([0.05, 0.1, 0.15])
            case 'LPA':
                net = LPA(N, m, gamma)
                if m == 1:
                    r_coeff = 0.01 + 0.04 * np.random.rand() if switch_type == 0 else \
                              switch_type * np.random.choice([0.05, 0.1, 0.15])
            case _:
                raise 'Topology not valid!!'

        ordering = 'deg' if np.random.rand() <= 0.5 else 'rnd'
        node_order = np.random.permutation(net.vcount()) if ordering == 'rnd' else net.degree()

        switch_no = 0
        while True:
            net = switch(net, node_order, type=switch_type)
            switch_no += 1
            if net is False:
                print('Switching trials maxed out; generating new net...')
                net = ig.Graph(n=2)
                break
            if (switch_type * net.assortativity(node_order) > switch_type * r_coeff) or \
               (switch_type == 0 and np.abs(net.assortativity(node_order)) < r_coeff):
                break
            if switch_no > 100000:
                print('Taking too many switches...')
                net = ig.Graph(n=2)
                break

        if net.is_connected():
            print('r_final =', net.assortativity(node_order), ordering)
            net_dict[net_no] = {'adj': np.array(net.get_adjacency().data, dtype=bool)}
            net_dict[net_no]['info'] = {
                'topology': topology,
                'size': N,
                'mean_deg': np.mean(net.degree()),
                'assortativity': net.assortativity(node_order),
                'ordering': ordering,
                'switch_count': switch_no
            }
            if topology in ['LPA', 'Copy']:
                net_dict[net_no]['info']['gamma'] = gamma
            if net_no % 100 == 0:
                with open('switched_graphs.pkl', 'wb') as out_f:
                    pickle.dump(net_dict, out_f)
            break
        trial += 1

with open('switched_graphs.pkl', 'wb') as out_f:
    pickle.dump(net_dict, out_f)
