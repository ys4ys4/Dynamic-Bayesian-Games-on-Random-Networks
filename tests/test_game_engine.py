import numpy as np
import networkx as nx

from simulations.game_engine import BeliefEngine


def test_mc_social_llr_accepts_smaller_sample_size():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)

    engine = BeliefEngine(
        3,
        "ER",
        "bounded",
        q=0.8,
        graph=graph,
        rng=np.random.default_rng(0),
        p=0.3,
        m=64,
    )

    obs = np.array([1, 0], dtype=int)
    llr = engine._mc_social_llr(
        2,
        np.array([0, 1]),
        obs,
        np.array([0, 1, 0], dtype=int),
    )

    assert engine.M == 64
    assert np.isfinite(llr)
