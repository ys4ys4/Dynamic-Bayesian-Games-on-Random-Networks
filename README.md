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

The environment is intentionally described by `pyproject.toml`; the local `venv/` directory is not a portable dependency specification.

## Smoke Test

Run a small deterministic simulation from the repository root:

```text
python -m simulations.smoke_test
```

This checks package imports, graph generation, the sequential engine, and the result container without starting any large experiment.

## Running Experiments

The `LS_*.py` modules contain the current large experiment entry points. They use the configurations in `simulations/experiment_configs.py` and write CSV output under `data/`. Review the agent count, run count, and output path before launching a long job, especially when transferring a run to HPC.

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
- Existing CSV and image files are generated artifacts and are ignored by Git. Preserve the exact inputs and commit separately when archiving a result.
- The notebooks are exploratory and may depend on an older API. The Python modules and the smoke test are the canonical runnable path.