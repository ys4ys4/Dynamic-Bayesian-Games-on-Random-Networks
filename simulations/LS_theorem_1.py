import time
import os

from simulations.experiment_configs import THEOREM_1_CONFIG
from simulations.run_experiments import (
    make_simple_progress_logger,
    run_overnight_sim,
)

config = THEOREM_1_CONFIG

print(f"Starting {config.name} simulation...")
start_time = time.time()

log_progress = make_simple_progress_logger(
    config.progress_axis_name,
    config.result_column_name,
    value_format="d"
)

run_overnight_sim(
    graph_type=config.graph_type,
    signal_type=config.signal_type,
    loop_values=config.loop_values,
    loop_param_name=config.loop_param_name,
    result_column_name=config.result_column_name,
    runs=config.runs,
    agents=config.agents,
    seed_base=config.seed_base,
    output_csv=config.output_csv_path,
    reducer=config.reducer,
    progress_callback=log_progress
)

end_time = time.time()
elapsed_hours = (end_time - start_time) / 3600
print(f"Completed {config.name} in {elapsed_hours:.2f} hours.")
print(f"Saved summary CSV to: {os.path.abspath(config.output_csv_path)}")
