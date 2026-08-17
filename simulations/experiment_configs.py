from dataclasses import dataclass

import numpy as np

from simulations.run_experiments import (
    summarise_cascade_metrics,
    collect_final_accuracies,
    make_simple_progress_logger,
    make_cascade_progress_logger
)


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    graph_type: str
    signal_type: str
    agents: int
    runs: int
    seed_base: int
    loop_values: np.ndarray
    loop_param_name: str
    result_column_name: str
    reducer: object
    progress_callback: object
    output_csv_path: str
    plot_path: str | None = None
    plot_title: str | None = None
    plot_xlabel: str | None = None
    plot_ylabel: str | None = None
    plot_figsize: tuple[int, int] | None = None
    plot_bins: tuple[int, int] | list[int] | None = None
    plot_cmap: str | None = None
    plot_cmin: int | None = None
    plot_dpi: int | None = None


# naive estimate = 9.38 hours with 1000 runs, 1000 agents, and 50 p vals
# MC estimate (tbd)
ER_BOUNDED_CONFIG = ExperimentConfig(
    name="ER bounded",
    graph_type="ER",
    signal_type="bounded",
    agents=1000,
    runs=1000,
    seed_base=42,
    loop_values=np.linspace(0.02, 1.0, 50),
    loop_param_name="p",
    result_column_name="p_value",
    reducer=summarise_cascade_metrics,
    output_csv_path="data/ER_bounded.csv",
    progress_callback=make_cascade_progress_logger(
        param="p",
        value_key="p_value"
    ),
    plot_path="data/ER_bounded_summary.png",
    plot_title="ER Bounded Summary",
    plot_xlabel="Probability of node connection (p)",
    plot_ylabel="Outcome probability / mean time",
    plot_figsize=(12, 10),
    plot_dpi=300
)

# estimated time = 12.25 hours with 1000 runs, 5000 agents, and 50 k vals
THEOREM_1_CONFIG = ExperimentConfig(
    name="theorem 1",
    graph_type="NEO",
    signal_type="unbounded",
    agents=5000,
    runs=1000,
    seed_base=42,
    loop_values=np.arange(1, 51),
    loop_param_name="k",
    result_column_name="EIAs",
    reducer=collect_final_accuracies,
    output_csv_path="data/Theorem_1_NEO_unbounded.csv",
    progress_callback=make_simple_progress_logger(
        param="k",
        value_key="EIAs",
        value_format="d"
    ),
    plot_path="data/Theorem_1_NEO_unbounded_heatmap.png",
    plot_title="Theorem 1 (Non-Expanding Observations with Unbounded Signals)",
    plot_xlabel="Number of Excessively Influential Agents (k)",
    plot_ylabel="Final Network Accuracy",
    plot_figsize=(12, 8),
    plot_bins=(50, 100),
    plot_cmap="viridis",
    plot_cmin=1,
    plot_dpi=300
)

# naive estimate = 8.53 hours with 1000 runs, 1000 agents, and 50 p vals
# MC estimate (tbd)
THEOREM_2_CONFIG = ExperimentConfig(
    name="theorem 2",
    graph_type="ER",
    signal_type="unbounded",
    agents=1000,
    runs=100,
    seed_base=42,
    loop_values=np.linspace(0.1, 1.0, 10),
    loop_param_name="p",
    result_column_name="p_value",
    reducer=summarise_cascade_metrics,
    output_csv_path="data/Theorem_2_ER_unbounded.csv",
    progress_callback=make_cascade_progress_logger(
        param="p",
        value_key="p_value"
    ),
    plot_path="data/Theorem_2_ER_unbounded_summary.png",
    plot_title="Theorem 2 (Erdös-Rényi Graphs with Unbounded Signals)",
    plot_xlabel="Probability of node connection (p)",
    plot_ylabel="Outcome probability / mean time",
    plot_figsize=(12, 10),
    plot_dpi=300
)

# estimated time = 6.89 hours with 1000 runs, 1000 agents, and 19 q vals
THEOREM_3I_CONFIG = ExperimentConfig(
    name="theorem 3i",
    graph_type="complete",
    signal_type="bounded",
    agents=1000,
    runs=1000,
    seed_base=42,
    loop_values=np.linspace(0.05, 0.95, 19),
    loop_param_name="q",
    result_column_name="signal_accuracy",
    reducer=summarise_cascade_metrics,
    output_csv_path="data/Theorem_3i_complete_bounded.csv",
    progress_callback=make_cascade_progress_logger(
        param="q",
        value_key="signal_accuracy"
    ),
    plot_path="data/Theorem_3i_complete_bounded_summary.png",
    plot_title="Theorem 3i (Complete Graphs with Bounded Signals)",
    plot_xlabel="Signal Accuracy",
    plot_ylabel="Outcome probability",
    plot_figsize=(12, 8),
    plot_dpi=300
)

# estimated time = 1.43 hours with 1000 runs, 1000 agents, and 19 q vals
THEOREM_3II_CONFIG = ExperimentConfig(
    name="theorem 3ii",
    graph_type="previous",
    signal_type="bounded",
    agents=5000,
    runs=1000,
    seed_base=42,
    loop_values=np.linspace(0.05, 0.95, 19),
    loop_param_name="q",
    result_column_name="signal_accuracy",
    reducer=collect_final_accuracies,
    output_csv_path="data/Theorem_3ii_previous_bounded.csv",
    progress_callback=make_simple_progress_logger(
        param="q",
        value_key="signal_accuracy"
    ),
    plot_path="data/Theorem_3ii_previous_bounded_heatmap.png",
    plot_title="Theorem 3ii (Dipath Graphs with Bounded Signals)",
    plot_xlabel="Signal Accuracy (q)",
    plot_ylabel="Final Network Accuracy",
    plot_figsize=(12, 8),
    plot_bins=(19, 100),
    plot_cmap="viridis",
    plot_cmin=1,
    plot_dpi=300
)

# naive estimate = 4.62 hours with 1000 runs, 10000 agents, and 19 q vals
# MC estimate (tbd)
THEOREM_3III_CONFIG = ExperimentConfig(
    name="theorem 3iii",
    graph_type="BS",
    signal_type="bounded",
    agents=10000,
    runs=1000,
    seed_base=42,
    loop_values=np.linspace(0.05, 0.95, 19),
    loop_param_name="q",
    result_column_name="signal_accuracy",
    reducer=collect_final_accuracies,
    output_csv_path="data/Theorem_3iii_BS_bounded.csv",
    progress_callback=make_simple_progress_logger(
        param="q",
        value_key="signal_accuracy"
    ),
    plot_path="data/Theorem_3iii_previous_bounded_heatmap.png",
    plot_title="Theorem 3iii (Bounded Sample Graphs with Bounded Signals)",
    plot_xlabel="Signal Accuracy (q)",
    plot_ylabel="Final Network Accuracy",
    plot_figsize=(12, 8),
    plot_bins=(19, 100),
    plot_cmap="viridis",
    plot_cmin=1,
    plot_dpi=300
)
