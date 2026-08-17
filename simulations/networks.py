# Graph generation

import networkx as nx
import numpy as np


# Basic non-expanding observations

def gen_neog(N, k, rng=None):
    """
    generates non-expanding observations graph (for theorem 1)
    agents after the first k EIAs only observe the first k agents
    the EIAs observe no one
    """
    rng = np.random.default_rng() if rng is None else rng
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    for i in range(k, N):
        for j in range(k):
            if rng.random() < 0.5:
                graph.add_edge(i, j)
    return graph


# Erdős-Rényi

def gen_erg(N, p, rng=None):
    """
    sequential Erdős-Rényi graph (for theorem 2)
    each agent sees each predecessor with probability p
    """
    rng = np.random.default_rng() if rng is None else rng
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))

    if p > 0:
        adj_matrix = rng.random((N, N)) < p
        lower_triangle = np.tril(adj_matrix, k=-1)
        i_indices, j_indices = np.nonzero(lower_triangle)
        graph.add_edges_from(zip(i_indices, j_indices))

    return graph


# Complete

def gen_complete(N):
    """
    complete graph (for theorem 3i)
    every agent sees every predecessor
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    for i in range(1, N):
        for j in range(i):
            graph.add_edge(i, j)
    return graph


# Previous

def gen_prev(N):
    """
    immediate predecessor graph (for theorem 3ii)
    agent i only sees agent i-1
    """
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))
    for i in range(1, N):
        graph.add_edge(i, i - 1)
    return graph


# Bounded sample

def gen_bounded_sample(N, k, rng=None):
    """
    stochastic bounded neighborhood (for theorem 3iii)
    each agent i observes exactly k predecessors chosen uniformly at random
    """
    rng = np.random.default_rng() if rng is None else rng
    graph = nx.DiGraph()
    graph.add_nodes_from(range(N))

    for i in range(1, N):
        num_to_observe = min(i, k)
        observed_agents = rng.choice(i, size=num_to_observe, replace=False)
        for j in observed_agents:
            graph.add_edge(i, j)

    return graph


# Uniform attachment - unused

# def gen_unif(N, rng=None):
#     """
#     uniform attachment model
#     """
#     rng = np.random.default_rng() if rng is None else rng
#     graph = nx.DiGraph()
#     graph.add_nodes_from(range(N))
#     for i in range(1, N):
#         graph.add_edge(i, rng.integers(0, i))
#     return graph
