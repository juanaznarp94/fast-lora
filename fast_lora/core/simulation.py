import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import cdist
from scipy.special import erf

from functools import cached_property

from .configs import (
    EndDeviceConfig,
    GatewayConfig,
    CommunicationConfig,
    LogDistancePathLossConfig,
)


class LoRaNetwork:
    _REQUIRED_SNR = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0])
    """Required SNR at gateway for SF from 7 to 12 for successfull packet reception.
    Final decision is based purely on RSS and gateway sensitivity.
    """

    _INTER_SF_INTERFERENCE = np.array(
        [
            [6, -8, -9, -9, -9, -9],
            [-11, 6, -11, -12, -13, -13],
            [-15, -13, 6, -13, -14, -15],
            [-19, -18, -17, 6, -17, -18],
            [-22, -22, -21, -20, 6, -20],
            [-25, -25, -25, -24, -23, 6],
        ]
    )
    """SIR thresholds for interfering end devices, where the rows
    indicate the suffering end device and the columns indicate the
    interfering end device.
    """

    _ALMOST_ZERO = 1e-10
    """Replacement value for values that should be zero to prevent division by 0"""

    def __init__(
        self,
        end_devices: EndDeviceConfig,
        gateways: GatewayConfig,
        communication_config: CommunicationConfig,
        log_distance_path_loss_config: LogDistancePathLossConfig,
        seed: int = 42,
    ):
        self._end_devices = end_devices
        self._gateways = gateways
        self._communication_config = communication_config
        self._log_distance_path_loss_config = log_distance_path_loss_config

        # initialize RNG
        self.rng = np.random.default_rng(seed)

    @property
    def end_devices(self) -> EndDeviceConfig:
        """The end devices in the network and their configuration."""
        return self._end_devices

    @property
    def gateways(self) -> GatewayConfig:
        """The gateways in the network and their configuration."""
        return self._gateways

    @property
    def communication_config(self) -> CommunicationConfig:
        """Various communication settings shared across all end devices."""
        return self._communication_config

    @property
    def log_distance_path_loss_config(self) -> LogDistancePathLossConfig:
        """Configuration parameters of the log distance path loss model."""
        return self._log_distance_path_loss_config

    @cached_property
    def distances(self) -> np.ndarray:
        """Distance between end devices (dim 0) and gateways (dim 1) in meters."""
        return cdist(self.end_devices.positions, self.gateways.positions)

    @property
    def rss(self) -> np.ndarray:
        """Expected RSS  in dBm of packet from end device (dim 0) arriving at
        gateway (dim 1) without considering any randomness in path loss.
        """
        return (
            # transmission power of end device
            self.end_devices.transmission_powers[:, np.newaxis]
            # reference path loss
            - self.log_distance_path_loss_config.reference_path_loss
            # actual path loss considering distance between end device and gateway
            - 10
            * self.log_distance_path_loss_config.path_loss_exponent
            * np.log10(
                np.maximum(self.distances, self._ALMOST_ZERO)
                / self.log_distance_path_loss_config.reference_distance
            )
        )

    @property
    def rss_sampled(self) -> np.ndarray:
        """RSS in dBm of specific packet from end device (dim 0) arriving at
        gateway (dim 1) with considered randomness in path loss and interference from
        other end devices. Packets that arrived with an RSS below the gateway sensitivity
        or that were corrupted by a transmission from another end device are represented
        with a value of `-np.inf`.
        """
        # apply attenuation by flat fading to RSS
        # see formula (6)
        rss_sampled = self.rss - self.rng.normal(
            0, self.log_distance_path_loss_config.flat_fading, size=self.rss.shape
        )
        # remove all values that are below the gateway sensitivity for the used SF
        rss_sampled[
            ~(
                rss_sampled
                >= self.gateways.sensitivity[self.end_devices.spreading_factors - 7][
                    :, np.newaxis
                ]
            )
        ] = -np.inf
        # remove values according to the probability of interference with other transmissions
        rss_sampled[
            ~(self.rng.random(self.prob_not_corrupted.shape) <= self.prob_not_corrupted)
        ] = -np.inf
        return rss_sampled

    @property
    def snr(self) -> np.ndarray:
        """Expected SNR in dB of packet from end device (dim 0) arriving at
        gateway (dim 1) without considering any randomness in path loss.

        Apply 'calculate_snr()' to 'rss_sampled' to calculated SNR of previously
        sampled transmissions.
        """
        return self.calculate_snr(self.rss)

    @property
    def snr_sampled(self) -> np.ndarray:
        """SNR in dB of specific packet from end device (dim 0) arriving at
        gateway (dim 1) with considered randomness in path loss and interference from
        other end devices. Packets that arrived with an RSS below the gateway sensitivity
        or that were corrupted by a transmission from another end device are represented
        with a value of `-np.inf`.
        """

    def calculate_snr(self, rss: np.ndarray) -> np.ndarray:
        """SNR of specific packet from end device (dim 0) arriving at gateway (dim 1) with considered flat fading and
        interference from other end devices, based on given RSS values previously sampled via `LoraModel.rss`. Packets
        that arrived with an RSS below the gateway sensitivity or that were interfered by a transmission from another
        end device are represented with a value of `-np.inf`."""
        return rss - self._thermal_noise

    @property
    def _thermal_noise(self) -> float:
        """The intensity of the background thermal noise in dBm. Automatically calculated
        such that SNR of transmissions that just failed is as close as possible to the
        above values. Reception decision is based purely on RSS and gateway sensitivy.
        """
        return np.mean(self.gateways.sensitivity - self._REQUIRED_SNR)

    @property
    def prob_exceeds_gateway_sensitivity(self) -> np.ndarray:
        """Probability of a packet from end device (dim 0) arriving at gateway (dim 1) to exceed the gateway sensitivity."""
        return 0.5 + 0.5 * erf(
            # expected RSS of packet from end device arriving at gateway - minimum RSS at gateway for spreading factor of end device
            (
                self.rss
                - self.gateways.sensitivity[self.end_devices.spreading_factors - 7][
                    :, np.newaxis
                ]
            )
            / (np.sqrt(2) * self.log_distance_path_loss_config.flat_fading)
        )

    @property
    def transmission_time(self) -> np.ndarray:
        """Time for an end device (dim 0) to transmit a full packet."""
        preamble_time = (
            self._communication_config.preamble_length + 4.25
        ) * self.symbol_time

        low_data_rate = np.zeros_like(self.end_devices.spreading_factors)
        if self.communication_config.low_data_rate is not None:
            low_data_rate[
                (
                    self.end_devices.spreading_factors
                    >= self.communication_config.low_data_rate[0]
                )
                & (
                    self.end_devices.spreading_factors
                    <= self._communication_config.low_data_rate[1]
                )
            ] = 1

        payload_time = (
            8
            + np.maximum(
                np.ceil(
                    (
                        self.communication_config.payload_size
                        - 4 * self.end_devices.spreading_factors
                        + 28
                        + 16
                    )
                    / (4 * (self.end_devices.spreading_factors - 2 * low_data_rate))
                )
                * (self.communication_config.coding_rate + 4),
                0,
            )
        ) * self.symbol_time

        return preamble_time + payload_time

    @property
    def symbol_time(self) -> np.ndarray:
        """Time for an end device (dim 0) to transmit a single symbol."""
        return np.power(2, self.end_devices.spreading_factors) / (
            self._communication_config.bandwidth
        )

    @property
    def prob_not_corrupted(self) -> np.ndarray:
        """Probability of a packet from end device (dim 0) arriving at
        gateway (dim 1) without being corrupted by a simultaneous
        transmission from any other end device.
        """
        # time interval during which an end device j (dim 1) would interfer with a transmitting end device i (dim 0)
        # considering the possible overlap (as preamble can be corrupted up to a certain point)
        interference_interval = (
            self.transmission_time
            - (self.communication_config.preamble_length - 5) * self.symbol_time
        ).reshape(-1, 1) + self.transmission_time
        # an end device cannot interfere with itself
        np.fill_diagonal(interference_interval, 0)

        # probability of an end device j (dim 1) transmitting at the same time as end device i (dim 0)
        prob_interference_transmission = 1 - np.exp(
            (-1)
            * (1 / self.communication_config.packet_interval)
            * interference_interval
        )

        # probability of a simultaneous transmission from an end device j (dim 1) to gateway k (dim 2)
        # being strong enough to interfere with a transmission from end device i (dim 0) to gateway k (dim 2)
        prob_interference_rss = 0.5 + 0.5 * erf(
            (
                # SIR threshold for transmitting end device i (dim 0) and interfering end device j (dim 1)
                self._INTER_SF_INTERFERENCE[
                    self.end_devices.spreading_factors - 7,
                    (self.end_devices.spreading_factors - 7).reshape(-1, 1),
                ][:, :, np.newaxis]
                # expected RSS difference between transmissions from end device i (dim 0) and j (dim 1)
                # at gateway k (dim 2)
                - (self.rss[:, np.newaxis, :] - self.rss[np.newaxis, :, :])
            )
            / (2 * np.sqrt(2) * self.FLAT_FADING)
        )

        # combined probability for no end device j to transmit at the same time as end device i (dim 0) to
        # gateway k (dim 1) while at the same time having a strong enough signal to corrupt
        # the transmission from end device i (dim 0) at gateway k (dim 1)
        return np.prod(
            1
            - prob_interference_transmission[:, :, np.newaxis] * prob_interference_rss,
            axis=1,
        )

    @property
    def pdr_per_gateway(self) -> np.ndarray:
        """Packet delivery ratio from end device (dim 0) to gateway (dim 1)."""
        return self.prob_exceeds_gateway_sensitivity * self.prob_not_corrupted

    @property
    def pdr(self) -> np.ndarray:
        """Packet delivery ratio from end device (dim 0) to any gateway."""
        return 1 - np.prod(1 - self.pdr_per_gateway, axis=1)

    @property
    def pdr_max(self) -> np.ndarray:
        """Estimated maximum achievable packet delivery ratio of an end
        device (dim 0) based on its distance to the closest gateway, not
        considering any packet interference.
        """
        return np.vectorize(self._calculate_max_pdr)(np.min(self.distances, axis=1))

    @property
    def ee(self) -> np.ndarray:
        """Energy efficiency in bits/mJ of an end device (dim 0) as bits
        successfully transmitted per energy consumed.
        """
        # see formula (11)
        return self.communication_config.payload_size / (
            np.vectorize(self.end_devices.transmission_power_consumption)(
                self.end_devices.transmission_powers
            )
            * self.transmission_time
            / np.maximum(self.pdr, self._ALMOST_ZERO)
        )
