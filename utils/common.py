import pickle

def load_g(path, name):
    with open(path, "rb") as f:
        g = pickle.load(f)
    g.simplify(multiple=True, loops=True)
    g['name'] = name
    return g