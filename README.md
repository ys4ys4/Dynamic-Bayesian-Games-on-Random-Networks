# Social Learning Simulations

Python simulations for studying Bayesian social learning on directed networks. The repository contains two related models:

- **Sequential Learning Model (SLM):** agents act once, in order, and observe actions from earlier agents.
- **Repeated Learning Model (RLM):** agents update and act over several rounds while observing connected neighbours.

The simulations support computational investigations related to Acemoglu et al. (2011) and Gale and Kariv (2003).

## Setup

Use Python 3.10 or newer. From the repository root:

```text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools:

```text
python -m pip install -e '.[dev]'
```

The environment is intentionally described by `pyproject.toml`; local virtual-environment directories are not portable dependency specifications.

## Smoke Test

Run a small deterministic simulation from the repository root:

```text
python -m simulations.smoke_test
```

This checks package imports, graph generation, the sequential engine, and the result container without starting any large experiment.

## Models and Networks

`SequentialGame` supports the following graph types:

| Graph type | Description | Signal types |
| --- | --- | --- |
| `NEO` | Non-expanding observations with `k` influential early agents | bounded, unbounded |
| `ER` | Sequential Erdős-Rényi graph with predecessor edge probability `p` | bounded, unbounded |
| `complete` | Each agent observes every predecessor | bounded, unbounded |
| `previous` | Each agent observes only the immediate predecessor | bounded, unbounded |
| `BS` | Each agent observes a bounded random sample of predecessors | bounded, unbounded |

`RepeatedGame` supports `complete_connected`, `connected_star`, and `dictator_star`. The repeated model currently uses unbounded Gaussian signals; bounded signals are rejected by the validation layer.

## Running Experiments

Run experiment commands from the repository root. The `LS_*.py` modules contain the large sequential-learning entry points. They use configurations from `simulations/experiment_configs.py` and write CSV output under `data/`.

For example:

```text
python -m simulations.LS_theorem_3i
```

The current sequential entry points are:

```text
python -m simulations.LS_ER_bounded
python -m simulations.LS_theorem_1
python -m simulations.LS_theorem_2
python -m simulations.LS_theorem_3i
python -m simulations.LS_theorem_3ii
python -m simulations.LS_theorem_3iii
```

Review the agent count, run count, and output path before launching a long job, especially when transferring a run to HPC.

For a small direct run:

```python
from simulations.game_engine import SequentialGame
from simulations.run_experiments import run_sim

result = run_sim(
    game_type=SequentialGame,
    graph_type="complete",
    agents=100,
    runs=10,
    seed=42,
    signal_type="bounded",
    q=0.8,
)
```

## Reproducibility Notes

- Pass an explicit integer `seed` to `run_sim`.
- A run uses one NumPy generator for graph generation and gameplay, so changing graph construction changes the subsequent random stream.
- Record the commit, Python version, dependency versions, configuration values, and output file for published results.
- Newly generated CSV and image files are ignored by Git. Preserve published outputs together with their exact inputs and commit when archiving a result.
- The notebooks are exploratory and aligned with the current Python API. The Python modules and the smoke test are the canonical runnable path.

## Project Layout

- `simulations/game_engine.py`: sequential and repeated game engines plus belief updating.
- `simulations/networks.py`: graph generators for both models.
- `simulations/run_experiments.py`: reusable single-run and parameter-sweep helpers.
- `simulations/experiment_configs.py`: long-running experiment definitions.
- `simulations/LS_*.py`: command-line experiment entry points.
- `data/`: tracked example results and the default destination for generated output.
- `*_testing.ipynb` and `RLM_testing.ipynb`: exploratory analysis and development notebooks.

## Current Scope

- Sequential experiments cover non-expanding, Erdős-Rényi, complete, immediate-predecessor, and bounded-sample networks.
- Monte Carlo belief updating approximates social beliefs for the stochastic ER and bounded-sample sequential graphs.
- Repeated experiments include exact belief updates for the supported complete and star network variants.
- Research and exploratory notebooks may contain work that is not yet exposed through a packaged experiment entry point.
