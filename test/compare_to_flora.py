import os
import numpy as np
import pandas as pd
from itertools import product
from tqdm import tqdm
from time import time_ns

from tempfile import TemporaryDirectory

from .network_config import RandomNetworkConfig
from .flora import run_flora_sim, get_metrics


if __name__ == "__main__":
    simulations_per_config = 10
    num_end_devices = [10, 20, 50, 100, 200]
    num_gateways = [1, 2, 3, 4, 5]

    simulation_configs = list(product(num_end_devices, num_gateways))

    rng = np.random.default_rng(42)

    # prepare dataframe for storing results
    result_df = pd.DataFrame()

    for i, j in tqdm(simulation_configs, desc="Simulating networks"):
        for k in tqdm(range(simulations_per_config), desc="Iterations"):
            config = RandomNetworkConfig(
                num_end_devices=i,
                num_gateways=j,
                placement_area=(0, 1000),
                spreading_factor_range=(7, 12),
                transmission_power_range=(2, 16, 2),
                rng=rng,
            )

            # simulate network in FAST LoRa
            start_time = time_ns()
            network = config.to_fast_lora_network()
            pdr = network.pdr
            ee = network.ee
            end_time = time_ns()

            elapsed_time_fast_lora = end_time - start_time

            # simulate network in FLoRa
            elapsed_time_flora = -1
            try:
                with TemporaryDirectory() as tempdir:
                    inifile_path = os.path.join(tempdir, "config.ini")
                    config.to_inifile(inifile_path, simtime_days=7)

                    start_time = time_ns()
                    _, scalarfile, _ = run_flora_sim(inifile_path, tempdir, "result")
                    flora_pdr, flora_ee = get_metrics(scalarfile)
                    end_time = time_ns()
                    elapsed_time_flora = end_time - start_time
            except Exception as e:
                pass

            result_df = pd.concat(
                [
                    result_df,
                    pd.DataFrame(
                        [
                            {
                                "RUN_NO": (k % simulations_per_config) + 1,
                                "NUM_END_DEVICES": i,
                                "NUM_GATEWAYS": j,
                                "TIME_FAST_LORA": elapsed_time_fast_lora,
                                "TIME_FLORA": elapsed_time_flora,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
            result_df.to_csv("speed-comparison.csv")
