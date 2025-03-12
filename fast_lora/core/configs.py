import numpy as np

from typing import Tuple
from functools import cache


class CommunicationConfig:
    def __init__(
        self,
        bandwidth: float,
        preamble_length: int,
        payload_size: int,
        packet_interval: float,
        coding_rate: int,
        low_data_rate: Tuple[int, int] | None = None,
    ):
        self._bandwidth = bandwidth
        self._preamble_length = preamble_length
        self._payload_size = payload_size
        self._packet_interval = packet_interval
        self._coding_rate = coding_rate
        self._low_data_rate = low_data_rate

    @property
    def bandwidth(self) -> float:
        return self._bandwidth

    @property
    def preamble_length(self) -> int:
        return self._preamble_length

    @property
    def payload_size(self) -> int:
        return self._payload_size

    @property
    def packet_interval(self) -> float:
        return self._packet_interval

    @property
    def coding_rate(self) -> int:
        return self._coding_rate

    @property
    def low_data_rate(self) -> Tuple[int, int] | None:
        return self._low_data_rate


class EndDeviceConfig:
    def __init__(
        self,
        positions: np.ndarray,
        spreading_factors: np.ndarray | int = None,
        transmission_powers: np.ndarray | int = None,
    ):
        self._positions = positions

        if isinstance(spreading_factors, int):
            self._spreading_factors = np.full(
                self.positions.shape[0], spreading_factors
            )
        else:
            self._spreading_factors = spreading_factors

        if isinstance(transmission_powers, int):
            self._transmission_powers = np.full(
                self._positions.shape[0], transmission_powers
            )
        else:
            self._transmission_powers = transmission_powers

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    @property
    def spreading_factors(self) -> np.ndarray:
        return self._spreading_factors

    @property
    def transmission_powers(self) -> np.ndarray:
        return self._transmission_powers

    @classmethod
    @cache
    def transmission_power_consumption(cls, tp: int) -> float:
        """Actual power consumption for a given LoRa transmission power setting in mW.

        - Calculation based on: Known and Unknown Facts of LoRa: Experiences from a Large-scale Measurement Study (Liando, et al. 2019)
        - Source: https://dl.acm.org/doi/pdf/10.1145/3293534, figure 8
        - Remark:
            - Formula (8) contains a polynomial function obtained through regression but weights given in paper seem incorrect
            - Weights below derived from figure 8 and polynomial regression to degree 10 for power consumption with transmission power settings between 2 and 16dBm (R² = 0.99997)
            - Assumed SX1276 chipset
        """
        weights = np.array(
            [
                1.89833380e-01,
                -1.35774139e-01,
                1.09830672e-01,
                -4.81384654e-02,
                1.32047168e-02,
                -2.34632756e-03,
                2.73338912e-04,
                -2.06703847e-05,
                9.75585755e-07,
                -2.60845237e-08,
                3.01521308e-10,
            ]
        )

        return np.polynomial.Polynomial(weights)(tp) * 1000


class GatewayConfig:
    def __init__(self, positions: np.ndarray, sensitivity: np.ndarray):
        self._positions = positions
        self._sensitivity = sensitivity

    @property
    def positions(self):
        return self._positions

    @property
    def sensitivity(self):
        return self._sensitivity


class LogDistancePathLossConfig:
    def __init__(
        self,
        reference_path_loss: float,
        path_loss_exponent: float,
        reference_distance: float,
        flat_fading: float,
    ):
        self.reference_path_loss = reference_path_loss
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance = reference_distance
        self.flat_fading = flat_fading
