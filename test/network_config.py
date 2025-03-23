from typing import Tuple
import numpy as np
from datetime import datetime

import fast_lora

import os
from string import Template

dirname = os.path.dirname(__file__)
TEMPLATE_PATH = os.path.join(dirname, "template.ini")

FLORA_PATH = os.environ.get("FLORA_PATH", "")


class RandomNetworkConfig:
    PACKET_GENERATION_INTERVAL = 1000

    def __init__(
        self,
        num_end_devices: int,
        num_gateways: int,
        placement_area: Tuple[int, int],
        spreading_factor_range: Tuple[int, int],
        transmission_power_range: Tuple[int, int, int],
        rng: np.random.Generator = None,
    ):
        # initialize RNG if needed
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng(int(datetime.now().timestamp()))

        # place end devices and gateways
        self.num_end_devices = num_end_devices
        self.num_gateways = num_gateways

        self.end_device_positions = self.rng.uniform(
            *placement_area, size=(self.num_end_devices, 2)
        )
        self.gateway_positions = self.rng.uniform(
            *placement_area, size=(self.num_gateways, 2)
        )

        # determine spreading factor and transmission power settings
        self.permitted_spreading_factors = np.arange(
            start=spreading_factor_range[0], stop=spreading_factor_range[1] + 1, step=1
        )
        self.permitted_transmission_powers = np.arange(
            start=transmission_power_range[0],
            stop=transmission_power_range[1] + 1,
            step=transmission_power_range[2],
        )

        self.spreading_factors = self.rng.choice(
            self.permitted_spreading_factors, size=(num_end_devices,)
        )
        self.transmission_powers = self.rng.choice(
            self.permitted_transmission_powers, size=(num_end_devices,)
        )

    def to_fast_lora_network(self) -> fast_lora.LoRaNetwork:
        # setup partial configurations
        communication_config = fast_lora.CommunicationConfig(
            bandwidth=125000,
            preamble_length=8,
            payload_size=10,
            packet_interval=self.PACKET_GENERATION_INTERVAL,
            coding_rate=1,
            allowed_spreading_factors=self.permitted_spreading_factors,
            allowed_transmission_powers=self.permitted_transmission_powers,
        )
        log_distance_path_loss_config = fast_lora.LogDistancePathLossConfig(
            reference_path_loss=127.41,
            path_loss_exponent=2.08,
            reference_distance=40,
            flat_fading=3.57,
        )
        end_device_config = fast_lora.EndDeviceConfig(
            positions=self.end_device_positions,
            spreading_factors=self.spreading_factors,
            transmission_powers=self.transmission_powers,
        )
        gateway_config = fast_lora.GatewayConfig(
            positions=self.gateway_positions,
            sensitivity=np.array([-124, -127, -130, -133, -135, -137]),
        )
        # create LoRa network to simulate
        network = fast_lora.LoRaNetwork(
            end_device_config,
            gateway_config,
            communication_config,
            log_distance_path_loss_config,
            seed=42,
        )
        return network

    def to_inifile(
        self,
        filepath: os.PathLike,
        simtime_days: int = 1,
    ):
        # load template inifile
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as file:
            content = "".join(file.readlines())
        template = Template(content)

        # prepare network features for insertion into template file
        node_features = self._features_to_string(
            np.concatenate(
                (
                    np.arange(self.num_end_devices).reshape(-1, 1),
                    self.end_device_positions,
                    self.transmission_powers.reshape(-1, 1),
                    self.spreading_factors.reshape(-1, 1),
                ),
                axis=1,
            ),
            self._inifile_node_features,
        )
        gateway_features = self._features_to_string(
            np.concatenate(
                (
                    np.arange(self.num_gateways).reshape(-1, 1),
                    self.gateway_positions,
                ),
                axis=1,
            ),
            self._inifile_gateway_features,
        )

        # replace sections in template
        output = template.safe_substitute(
            {
                "enable_adr": "false",
                "simtime_days": f"{simtime_days}d",
                "packet_generation_seconds": f"{self.PACKET_GENERATION_INTERVAL}s",
                "number_of_nodes": self.num_end_devices,
                "number_of_gateways": self.num_gateways,
                "node_features": node_features,
                "gateway_features": gateway_features,
                "energy_consumption_parameter_file": os.path.join(
                    FLORA_PATH, "simulations", "energyConsumptionParameters.xml"
                ),
                "cloud_delay_file": os.path.join(
                    FLORA_PATH, "simulations", "cloudDelays.xml"
                ),
            }
        )

        # create actual inifile
        if os.path.dirname(filepath) != "":
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as file:
            file.write(output)

    @classmethod
    def _inifile_node_features(
        cls, idx: int, pos_x: float, pos_y: float, initial_tp: int, initial_sf: int
    ) -> str:
        idx = int(idx)
        return "\n".join(
            [
                f"**.loRaNodes[{idx}].**.initialX = {pos_x:.2f}m",
                f"**.loRaNodes[{idx}].**.initialY = {pos_y:.2f}m",
                f"**.loRaNodes[{idx}].**initialLoRaSF = {int(initial_sf)}",
                f"**.loRaNodes[{idx}].**initialLoRaTP = {int(initial_tp)}dBm",
                f"**.loRaNodes[{idx}].**initialLoRaBW = 125 kHz",
                f"**.loRaNodes[{idx}].**initialLoRaCR = 1",
                f"**.loRaNodes[{idx}].**dataSize = 10B",
            ]
        )

    @classmethod
    def _inifile_gateway_features(cls, idx: int, pos_x: float, pos_y: float) -> str:
        idx = int(idx)
        return "\n".join(
            [
                f"**.loRaGW[{idx}].**.initialX = {pos_x:.2f}m",
                f"**.loRaGW[{idx}].**.initialY = {pos_y:.2f}m",
            ]
        )

    @classmethod
    def _features_to_string(cls, features: np.ndarray, func: callable) -> str:
        return "\n".join(list(map(lambda x: func(*x), features.tolist())))


if __name__ == "__main__":
    generator = RandomNetworkConfig(
        10, 3, (0, 1000), (7, 12), (2, 16, 2), np.random.default_rng(42)
    )
    network = generator.to_fast_lora_network()
    generator.to_inifile("test.ini")
