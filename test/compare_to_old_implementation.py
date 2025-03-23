import fast_lora
import numpy as np
from tqdm import tqdm
from itertools import product

from .old_implementation import LoraNetworkSimulation


# constant configuration - not influenced by number of end devices/gateways

DEFAULT_COMMUNICATION_CONFIG = fast_lora.CommunicationConfig(
    bandwidth=LoraNetworkSimulation.BANDWIDTH * 1000,
    preamble_length=LoraNetworkSimulation.PREAMBLE_LEN,
    payload_size=LoraNetworkSimulation.PAYLOAD_SIZE,
    packet_interval=LoraNetworkSimulation.PACKET_INTERVAL,
    coding_rate=LoraNetworkSimulation.CODING_RATE,
    allowed_spreading_factors=LoraNetworkSimulation.ALLOWED_SFS,
    allowed_transmission_powers=LoraNetworkSimulation.ALLOWED_TPS,
    low_data_rate=(11, 12),
)

DEFAULT_LOG_DISTANCE_PATH_LOSS_CONFIG = fast_lora.LogDistancePathLossConfig(
    reference_path_loss=LoraNetworkSimulation.PATH_LOSS_REF,
    path_loss_exponent=LoraNetworkSimulation.PATH_LOSS_EXP,
    reference_distance=LoraNetworkSimulation.REF_DIST,
    flat_fading=LoraNetworkSimulation.FLAT_FADING,
)

DEFAULT_GATEWAY_SENSITIVITY = LoraNetworkSimulation.GATEWAY_SENSITIVITY

MIN_COORDINATES = 0
MAX_COORDINATES = 1000


def compare_single_network(num_end_devices: int, num_gateways: int, seed: int):
    rng = np.random.default_rng(seed)

    # determine placement, spreading factor and transmission power settings
    end_device_positions = rng.uniform(
        low=MIN_COORDINATES, high=MAX_COORDINATES, size=(num_end_devices, 2)
    )
    gateway_positions = rng.uniform(
        low=MIN_COORDINATES, high=MAX_COORDINATES, size=(num_gateways, 2)
    )
    spreading_factors = rng.choice(
        np.arange(start=7, stop=12 + 1, step=1), size=(num_end_devices,)
    )
    transmission_powers = rng.choice(
        np.arange(start=2, stop=16 + 1, step=2), size=(num_end_devices,)
    )

    # setup new simulation
    end_devices = fast_lora.EndDeviceConfig(
        positions=end_device_positions,
        spreading_factors=spreading_factors,
        transmission_powers=transmission_powers,
    )
    gateways = fast_lora.GatewayConfig(
        positions=gateway_positions,
        sensitivity=DEFAULT_GATEWAY_SENSITIVITY,
    )
    network = fast_lora.LoRaNetwork(
        end_devices=end_devices,
        gateways=gateways,
        communication_config=DEFAULT_COMMUNICATION_CONFIG,
        log_distance_path_loss_config=DEFAULT_LOG_DISTANCE_PATH_LOSS_CONFIG,
        seed=seed,
    )

    # setup previous implementation
    previous_simulation = LoraNetworkSimulation(
        num_end_devices=num_end_devices,
        num_gateways=num_gateways,
        pos_end_device=end_device_positions,
        pos_gateway=gateway_positions,
        tp=transmission_powers,
        sf=spreading_factors,
        seed=seed,
    )

    # compare calculated metrics
    np.testing.assert_allclose(network.pdr_per_gateway, previous_simulation.pdr_gateway)
    np.testing.assert_allclose(network.pdr, previous_simulation.pdr)
    np.testing.assert_allclose(network.pdr_max, previous_simulation.max_pdr)
    np.testing.assert_allclose(network.ee, previous_simulation.ee)

    # compare expected transmission quality measurements
    np.testing.assert_allclose(network.rss, previous_simulation.rss_no_flat_fading)
    np.testing.assert_allclose(
        network.snr,
        previous_simulation.calculate_snr(previous_simulation.rss_no_flat_fading),
    )

    # compare sampled transmission quality measurements
    np.testing.assert_allclose(network.rss_sampled, previous_simulation.rss)
    np.testing.assert_allclose(
        network.calculate_snr(network.rss_sampled),
        previous_simulation.calculate_snr(previous_simulation.rss),
    )


if __name__ == "__main__":
    num_end_devices = np.round(np.logspace(start=1, stop=3, base=10, num=30)).astype(
        int
    )
    num_gateways = np.linspace(1, 5, 5).astype(int)

    seed = 0
    configs = list(product(num_end_devices, num_gateways))
    for i, j in tqdm(configs, desc="Comparing networks"):
        compare_single_network(i, j, seed)
        seed += 1
