# Social Learning Simulations

Python simulations for sequential and repeated Bayesian social-learning models, including the network structures used to study the results of Acemoglu et al. (2011) and Gale and Kariv (2003).

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

## Running Experiments

Run experiment commands from the repository root. The `LS_*.py` modules contain the current large sequential-learning entry points. They use the configurations in `simulations/experiment_configs.py` and write CSV output under `data/`.

For example:

```text
python -m simulations.LS_theorem_3i
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

## Current Scope

- The sequential-learning workflow is the currently validated simulation path.
- Monte Carlo belief updating is an approximation for the stochastic ER and bounded-sample network models.
- Repeated-learning work is ongoing. Complete-graph experiments are under development, and star-graph belief logic has not yet been implemented.
