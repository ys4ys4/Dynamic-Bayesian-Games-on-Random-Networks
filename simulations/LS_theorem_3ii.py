import time
import os

from simulations.experiment_configs import THEOREM_3II_CONFIG
from simulations.run_experiments import run_overnight_sim

config = THEOREM_3II_CONFIG

print(f"Starting {config.name} simulation...")
start_time = time.time()

run_overnight_sim(
    game_type=config.game_type,
    graph_type=config.graph_type,
    agents=config.agents,
    runs=config.runs,
    seed_base=config.seed_base,
    signal_type=config.signal_type,
    loop_param_name=config.loop_param_name,
    loop_values=config.loop_values,
    result_column_name=config.result_column_name,
    reducer=config.reducer,
    output_csv_path=config.output_csv_path,
    progress_callback=config.progress_callback,
    extra_params=config.extra_params
)

end_time = time.time()
elapsed_hours = (end_time - start_time) / 3600
print(f"Completed {config.name} in {elapsed_hours:.2f} hours.")
print(f"Saved summary CSV to: {os.path.abspath(config.output_csv_path)}")
