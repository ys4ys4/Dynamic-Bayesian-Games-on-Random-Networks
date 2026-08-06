import numbers
from dataclasses import dataclass
import pandas as pd
import numpy as np

from simulations.game_engine import SequentialGame
from simulations.networks import (
    gen_complete,
    gen_erg,
    gen_prev,
    gen_neog,
    gen_unif,
    gen_bounded_sample
)


@dataclass(frozen=True)
class SimulationResult:
    """
    container for the main outputs of a simulation run
    keeps result readable and explicit while supporting tuple-style unpacking
    """

    running_accuracies: list
    convergence_metrics: list
    params: dict

    def __iter__(self):
        yield self.running_accuracies
        yield self.convergence_metrics
        yield self.params


# large sim reducer

def summarize_cascade_metrics(convergence_metrics, total_runs):
    """
    summarises convergence metrics from large simulation
    returns a dictionary with the following keys:
    - prob_false_cascade: probability of a false cascade
    - prob_true_cascade: probability of a true cascade
    - prob_no_consensus: probability of no consensus
    - avg_success_time: average time to reach a true cascade (if any)
    - avg_fail_time: average time to reach a false cascade (if any)
    """

    false_cascades = 0
    true_cascades = 0
    success_time = 0
    fail_time = 0

    for converged, success, lock_time, _ in convergence_metrics:
        if converged:
            if success:
                true_cascades += 1
                success_time += lock_time
            else:
                false_cascades += 1
                fail_time += lock_time

    p_false = false_cascades / total_runs
    p_true = true_cascades / total_runs

    return {
        "prob_false_cascade": p_false,
        "prob_true_cascade": p_true,
        "prob_no_consensus": 1 - (p_false + p_true),
        "avg_success_time": (
            success_time / true_cascades if true_cascades else None
        ),
        "avg_fail_time": (
            fail_time / false_cascades if false_cascades else None
        )
    }


def collect_final_accuracies(convergence_metrics, _total_runs):
    """
    collects final accuracies from convergence metrics
    """
    return {
        "final_accuracies": [
            float(final_accuracy)
            for _, _, _, final_accuracy in convergence_metrics
        ]
    }


# large sim progress loggers

def make_simple_progress_logger(label, value_key, *, value_format=""):
    """
    returns simple progress logger function for large simulations
    logs value of loop parameter and result column
    """

    def log_progress(row):
        value = row[value_key]
        formatted_value = (
            format(value, value_format) if value_format else value
        )
        print(f"Finished {label}={formatted_value}.")

    return log_progress


def make_cascade_progress_logger(label, value_key):
    """
    returns more functional progress logger function for large simulations
    which logs:
    - loop parameter
    - probability of false cascade
    - probability of true cascade
    - average time to reach a true cascade (if any)
    - average time to reach a false cascade (if any)
    """

    def log_progress(row):
        success_time = (
            row["avg_success_time"]
            if row["avg_success_time"] is not None
            else "N/A"
        )
        fail_time = (
            row["avg_fail_time"]
            if row["avg_fail_time"] is not None
            else "N/A"
        )
        print(
            f"Finished {label}={row[value_key]:.2f} | "
            f"False Cascade: {row['prob_false_cascade']:.1%} | "
            f"True Cascade: {row['prob_true_cascade']:.1%} | "
            f"Avg Success Time: {success_time} | Avg Fail Time: {fail_time}"
        )

    return log_progress


# simulation functions

def _validate_run_sim_inputs(
    graph_type,
    signal_type,
    agents,
    p,
    k,
    q,
    runs
):
    """
    validates inputs for run_sim function
    """
    valid_graph_types = {"NEO", "ER", "complete", "UAM", "previous", "BS"}
    valid_signal_types = {"bounded", "unbounded"}

    if graph_type not in valid_graph_types:
        raise ValueError(
            "Invalid graph_type provided. Expected one of: "
            f"{sorted(valid_graph_types)}."
        )

    if signal_type not in valid_signal_types:
        raise ValueError(
            "Invalid signal_type provided. Expected one of: "
            f"{sorted(valid_signal_types)}."
        )

    if not isinstance(agents, numbers.Integral) or agents <= 0:
        raise ValueError("agents must be a positive integer.")

    if not isinstance(runs, numbers.Integral) or runs <= 0:
        raise ValueError("runs must be a positive integer.")

    if not isinstance(k, numbers.Integral) or k <= 0:
        raise ValueError("k (EIAs) must be a positive integer.")

    if graph_type == "ER" and not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1 for ER graphs.")

    if signal_type == "bounded":
        if not isinstance(q, numbers.Real) or not 0 < q < 1:
            raise ValueError("q must be a real number\
                              strictly between 0 and 1.")


def _get_graph_generator(
    graph_type,
    agents,
    p=0.05,
    k=1,
    sample=10,
    rng=None
):
    """
    returns a graph generator function based on the specified graph_type
    """
    if graph_type == "NEO":
        return lambda: gen_neog(agents, k)
    if graph_type == "ER":
        return lambda: gen_erg(agents, p, rng=rng)
    if graph_type == "complete":
        return lambda: gen_complete(agents)
    if graph_type == "UAM":
        return lambda: gen_unif(agents, rng=rng)
    if graph_type == "previous":
        return lambda: gen_prev(agents)
    if graph_type == "BS":
        return lambda: gen_bounded_sample(agents, sample, rng=rng)
    raise ValueError("Invalid graph_type provided.")


def run_sim(
    graph_type,
    signal_type,
    game_type=SequentialGame,
    agents=1000,
    p=0.05,
    k=1,
    q=0.8,
    runs=5,
    sample=10,
    seed=None
):
    """
    runs simulation with specified parameters -
    graph_type: type of graph to generate (NEO, ER, complete, previous, BS)
    signal_type: type of signal to use (bounded, unbounded)
    game_type: class of the game to run
    agents: number of agents in the simulation
    p: probability of connection for ER graphs (ignored for other graphs)
    k: number of EIAs for NEO graphs (ignored for other graphs)
    q: signal accuracy for bounded signals (ignored for unbounded signals)
    runs: number of simulation runs to perform
    sample: number of predecessors for BS graphs (ignored for other graphs)
    seed: random seed for reproducibility
    returns a SimulationResult
    """
    _validate_run_sim_inputs(
        graph_type=graph_type,
        signal_type=signal_type,
        agents=agents,
        p=p,
        k=k,
        q=q,
        runs=runs
    )

    rng = np.random.default_rng(seed)

    running_accuracies = []
    convergence_metrics = []
    params = {
        "graph_type": graph_type,
        "signal_type": signal_type,
        "agents": agents,
        "runs": runs,
        "seed": seed
    }

    if signal_type == "bounded":
        params["q"] = q

    if graph_type == "NEO":
        params["k"] = k
    elif graph_type == "ER":
        params["p"] = p
    elif graph_type == "BS":
        params["sample"] = sample

    gen_graph = _get_graph_generator(
        graph_type,
        agents,
        p=p,
        k=k,
        rng=rng,
        sample=sample
    )

    for _ in range(runs):
        graph = gen_graph()
        game = game_type(graph,
                         graph_type,
                         signal_type,
                         q, rng=rng,
                         k=k,
                         p=p,
                         sample=sample
                         )
        game.play()
        running_accuracies.append(game.running_accuracy())
        convergence_metrics.append(game.convergence_metrics())

    return SimulationResult(
        running_accuracies=running_accuracies,
        convergence_metrics=convergence_metrics,
        params=params
    )


def run_overnight_sim(
    *,
    graph_type,
    signal_type,
    loop_values,
    loop_param_name,
    result_column_name=None,
    runs,
    agents,
    seed_base=42,
    output_csv=None,
    reducer=summarize_cascade_metrics,
    extra_params=None,
    progress_callback=None
):
    """
    runs a large simulation with specified parameters -
    graph_type: type of graph to generate (NEO, ER, complete, previous, BS)
    signal_type: type of signal to use (bounded, unbounded)
    loop_values: list of values to loop over for specified parameter
    loop_param_name: name of parameter to loop over (e.g., "p", "k", "q")
    result_column_name: column to store loop parameter values in output CSV
    runs: number of simulation runs to perform for each loop value
    agents: number of agents in simulation
    seed_base: base random seed for reproducibility
    output_csv: path to save summary CSV
    reducer: function to summarize convergence metrics
    extra_params: dictionary of additional parameters to pass to run_sim
    progress_callback: function to log progress after each loop value
    returns a DataFrame with summary of results for each loop value
    """

    extra_params = extra_params or {}
    result_column_name = result_column_name or loop_param_name
    rows = []
    run_kwargs = {
        "graph_type": graph_type,
        "signal_type": signal_type,
        "agents": agents,
        "runs": runs,
        **extra_params
    }

    for i, value in enumerate(loop_values):
        run_kwargs["seed"] = seed_base + i * 1000
        run_kwargs[loop_param_name] = value

        result = run_sim(**run_kwargs)
        summary = reducer(result.convergence_metrics, runs)
        row = {result_column_name: value, **summary}
        rows.append(row)

        if output_csv:
            pd.DataFrame(rows).to_csv(output_csv, index=False)

        if progress_callback is not None:
            progress_callback(row)

    return pd.DataFrame(rows)
