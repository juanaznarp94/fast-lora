from functools import cache, cached_property

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import erf


class LoraNetworkSimulation:
    """Analytical Lora model derived from proposal in:
    Multiagent Reinforcement Learning with an Attention Mechanism for Improving Energy Efficiency in LoRa Networks

    - Author: Xu Zhang, et. al.
    - Year: 2023
    - Source: https://arxiv.org/pdf/2309.08965

    :param num_end_devices: Number of end devices to simulate
    :param num_gateways: Number of gateways to simulate
    :param pos_end_device: Array of shape (num_end_devices, 2) with the x and y coordinates of each end device in meters
    :param pos_gateway: Array of shape (num_gateways, 2) with the x and y coordinates of each gateway in meters
    :param tp: Array of shape (num_end_devices, 1) with the transmission power setting for each end device (see ALLOWED_TPS for allowed values)
    :param sf: Array of shape (num_end_devices, 1) with the spreading factor setting for each end device (see ALLOWED_SFS for allowed values)
    :param seed: Seed to initialize the internal random number generator (defaults to 42)
    """

    PATH_LOSS_REF = 127.41
    """Reference path loss in dB per `REF_DIST`.
    
    - Denoted as PL(d_0)
    - Reference for exact value: Do LoRa Low-Power Wide-Area Networks Scale?, Bor et al., 2016 (3rd paragraph after formula (3))
    - Source: https://www.link-labs.com/hubfs/DOCS.linklabs.com/2017/01/lora-scalability_r254.pdf
    - Note: As used by FLoRa simulation framework
    """
    REF_DIST = 40
    """Reference distance for `PATH_LOSS_REF` in m.
    
    - Denoted as d_0
    - Reference for exact value: Do LoRa Low-Power Wide-Area Networks Scale?, Bor et al., 2016 (3rd paragraph after formula (3))
    - Source: https://www.link-labs.com/hubfs/DOCS.linklabs.com/2017/01/lora-scalability_r254.pdf#
    - Note: As used by FLoRa simulation framework
    """
    PATH_LOSS_EXP = 2.08
    """Path loss exponent.
    
    - Denoted as &gamma;
    - Reference for exact value: Do LoRa Low-Power Wide-Area Networks Scale?, Bor et al., 2016 (3rd paragraph after formula (3))
    - Source: https://www.link-labs.com/hubfs/DOCS.linklabs.com/2017/01/lora-scalability_r254.pdf
    - Note: As used by FLoRa simulation framework
    """
    FLAT_FADING = 3.57
    """Standard deviation of Gaussian variable for calculating flat fading in path loss in dB.

    - Denoted as &sigma;
    - Reference for exact value: Do LoRa Low-Power Wide-Area Networks Scale?, Bor et al., 2016 (3rd paragraph after formula (3))
    - Source: https://www.link-labs.com/hubfs/DOCS.linklabs.com/2017/01/lora-scalability_r254.pdf
    - Note: As used by FLoRa simulation framework
    """

    GATEWAY_SENSITIVITY = np.array([-124, -127, -130, -133, -135, -137])
    """Min. RSS at gateway for SF from 7 to 12.
    
    - Denoted as &eta;_fi
    - Reference for exact value: Semtech SX1272/73 datasheet, table 10, Rev 3.1, March 2017
    - Source: https://semtech.my.salesforce.com/sfc/p/E0000000JelG/a/440000001NCE/v_VBhk1IolDgxwwnOpcS_vTFxPfSEPQbuneK3mWsXlU
    - Note: As used by FLoRa simulation framework
    """

    REQUIRED_SNR = np.array([-7.5, -10.0, -12.5, -15.0, -17.5, -20.0])
    """Required SNR at gateway for SF from 7 to 12

    Used to calculate background thermal noise for SNR calculation, as reception decision is based purely on RSS and gateway
    sensitivity. This behaviour is implemented to match the implementation in FLoRa as close as possible and permit its use as
    discrete simulation framework. Background thermal noise is calculated such that SNR of transmissions that just failed is
    as close as possible to the above values.

    - Source: https://www.thethingsnetwork.org/forum/uploads/default/original/2X/7/7480e044aa93a54a910dab8ef0adfb5f515d14a1.pdf
    """

    INTER_SF_INTERFERENCE = np.array(
        [
            [6, -8, -9, -9, -9, -9],
            [-11, 6, -11, -12, -13, -13],
            [-15, -13, 6, -13, -14, -15],
            [-19, -18, -17, 6, -17, -18],
            [-22, -22, -21, -20, 6, -20],
            [-25, -25, -25, -24, -23, 6],
        ]
    )
    """SIR thresholds for interfering end devices, where the rows indicate the suffering end device
    and the columns indicate the interfering end device.
    
    - Denoted as &omega;_ij
    - See TABLE I"""

    CODING_RATE = 1
    """Coding rate for LoRa, where the actual coding rate is calculated as 4/(4+`CODING_RATE`).
    Integer value between 1 and 4.
    
    - Denoted as CR
    """
    BANDWIDTH = 125
    """Bandwidth in kHz.
    
    - Denoted as B_C
    - Reference for exact value: Regional parameter specifications for EU863-870 band
    """

    PREAMBLE_LEN = 8
    """Number of preamble symbols.
    
    - Denoted as n_pr
    - Reference for exact value: Regional parameter specifications for EU863-870 band
    """
    PAYLOAD_SIZE = 80
    """Payload size in bits.
    
    - Denoted as L
    - Constantly assumed as 80 (variable payload length not considered for simulation)
    """

    PACKET_INTERVAL = 1000
    """Packet interval in s.
    
    - Used for calculation of &lambda; (packet generation rate) by 1/&lambda;
    - Constantly assumed as 1000s (variable package generation rate not considered for simulation)
    """

    ALLOWED_SFS = np.arange(7, 12 + 1)
    """Allowed values for setting the spreading factor."""
    ALLOWED_TPS = np.arange(2, 16 + 1, step=2)
    """Allowed values for setting the transmission power."""

    _ALMOST_ZERO = 1e-10
    """Replacement value for values that should be zero to prevent division by 0"""

    def __init__(
        self,
        num_end_devices: int,
        num_gateways: int,
        pos_end_device: np.ndarray,
        pos_gateway: np.ndarray,
        tp: np.ndarray,
        sf: np.ndarray,
        seed: int = 42,
    ):
        self._num_end_devices = num_end_devices
        self._num_gateways = num_gateways
        self._tp = tp
        self._sf = sf
        self._pos_end_device = pos_end_device
        self._pos_gateway = pos_gateway

        self.rng = np.random.default_rng(seed)

        # sanity check
        self._sanity_check()

        # clear cache
        self._invalidate_cache()

    def _sanity_check(self):
        """Checks attributes for validity."""
        if self.num_end_devices < 0:
            raise ValueError("num_end_devices must be greater than one")

        if self.num_gateways < 0:
            raise ValueError("num_gateways must be greater than one")

        if self.tp.shape != (self.num_end_devices,):
            raise ValueError("tp must be of shape (num_end_devices,)")

        if not np.all(np.in1d(self.tp, self.ALLOWED_TPS)):
            raise ValueError(
                f"tp must not contain values other than: {', '.join(map(str, self.ALLOWED_TPS))}"
            )

        if self.sf.shape != (self.num_end_devices,):
            raise ValueError("sf must be of shape (num_end_devices, )")

        if not np.all(np.in1d(self.sf, self.ALLOWED_SFS)):
            raise ValueError(
                f"sf must not contain values other than: {', '.join(map(str, self.ALLOWED_SFS))}"
            )

        if self.pos_end_device.shape != (self.num_end_devices, 2):
            raise ValueError("pos_end_device must be of shape (num_end_devices, 2)")

        if self.pos_gateway.shape != (self.num_gateways, 2):
            raise ValueError("pos_gateway must be of shape (num_gateways,2)")

    def _invalidate_cache(self):
        """Invalidates all cached properties and should be called whenever any attribute of the simulation is changed."""
        cached_properties = [
            "distance",
            "rss_no_flat_fading",
            "snr_no_flat_fading",
            "p_exceeds_gw_sensitivity",
            "transmission_and_symbol_time",
            "p_not_corrupted",
            "pdr_gateway",
            "pdr",
            "ee",
            "max_pdr",
        ]
        for prop in cached_properties:
            self.__dict__.pop(prop, None)

    @classmethod
    def _merge_or_replace(cls, old: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Tries to merge an old array with a new array, where the new array contains values other than np.nan.
        If the new array has a different shape than the old array, then it is completely replaced.
        """
        if not np.isnan(new).any():
            return new
        elif old.shape == new.shape:
            valid_indices = ~np.isnan(new)
            old[valid_indices] = new[valid_indices]
            return old
        else:
            raise ValueError(
                "Cannot merge arrays of different shapes if there are NaN values in the new array"
            )

    def _default_setter(self, attr_name, value):
        """Default setter for attributes that handles cache invalidation and sanity checking."""
        setattr(
            self,
            f"_{attr_name}",
            self._merge_or_replace(getattr(self, f"_{attr_name}"), value),
        )
        self._invalidate_cache()
        self._sanity_check()

    @property
    def num_end_devices(self) -> int:
        """The number of end devices in the simulation."""
        return self._num_end_devices

    @num_end_devices.setter
    def num_end_devices(self, num_end_devices: int):
        self._default_setter("num_end_devices", num_end_devices)

    @property
    def num_gateways(self) -> int:
        """The number of gateways in the simulation."""
        return self._num_gateways

    @num_gateways.setter
    def num_gateways(self, num_gateways: int):
        self._default_setter("num_gateways", num_gateways)

    @property
    def tp(self) -> np.ndarray:
        """The transmission power an end device (dim 0) uses for transmission."""
        return self._tp

    @tp.setter
    def tp(self, tp: np.ndarray):
        self._default_setter("tp", tp)
        self._tp = self._tp.astype(np.int32)

    @property
    def sf(self) -> np.ndarray:
        """The spreading factor an end device (dim 0) uses for transmission."""
        return self._sf

    @sf.setter
    def sf(self, sf: np.ndarray):
        self._default_setter("sf", sf)
        self._sf = self._sf.astype(np.int32)

    @property
    def pos_end_device(self) -> np.ndarray:
        """The position of an end device (dim 0) within the simulation in meters relative to an arbitrary origin (0, 0)."""
        return self._pos_end_device

    @pos_end_device.setter
    def pos_end_device(self, pos_end_device: np.ndarray):
        self._default_setter("pos_end_device", pos_end_device)

    @property
    def pos_gateway(self) -> np.ndarray:
        """The position of a gateway (dim 0) within the simulation in meters relative to an arbitrary origin (0, 0)."""
        return self._pos_gateway

    @pos_gateway.setter
    def pos_gateway(self, pos_gateway: np.ndarray):
        self._default_setter("pos_gateway", pos_gateway)

    @cached_property
    def distance(self) -> np.ndarray:
        """Distance between end devices (dim 0) and gateways (dim 1) in m.

        - Denoted as d_ik
        """
        return cdist(self._pos_end_device, self._pos_gateway)

    @cached_property
    def rss_no_flat_fading(self) -> np.ndarray:
        """Expected RSS of packet from end device (dim 0) arriving at gateway (dim 1) without considering attenuation by flat fading.

        - Denoted as z_ik
        """
        # see paragraph between formula (6) and (7)
        return (
            # transmission power of end device
            self._tp[:, np.newaxis]
            # reference path loss
            - self.PATH_LOSS_REF
            # actual path loss considering distance between end device and gateway
            - 10
            * self.PATH_LOSS_EXP
            * np.log10(np.maximum(self.distance, self._ALMOST_ZERO) / self.REF_DIST)
        )

    @cached_property
    def snr_no_flat_fading(self) -> np.ndarray:
        """Expected SNR of packet from end device (dim 0) arriving at gateway (dim 1) without considering attenuation by flat fading."""
        return self.calculate_snr(self.rss_no_flat_fading)

    @property
    def rss(self) -> np.ndarray:
        """RSS of specific packet from end device (dim 0) arriving at gateway (dim 1) with considered flat fading and
        interference from other end devices. Packets that arrived with an RSS below the gateway sensitivity or that were
        interfered by a transmission from another end device are represented with a value of `-np.inf`.

        - Denoted as RSS_ik
        """
        # apply attenuation by flat fading to RSS
        # see formula (6)
        actual_rss = self.rss_no_flat_fading - self.rng.normal(
            0, self.FLAT_FADING, size=self.rss_no_flat_fading.shape
        )
        # remove all values that are below the gateway sensitivity for the used SF
        actual_rss[
            ~(actual_rss >= self.GATEWAY_SENSITIVITY[self._sf - 7][:, np.newaxis])
        ] = -np.inf
        # remove values according to the probability of interference with other transmissions
        actual_rss[
            ~(self.rng.random(self.p_not_corrupted.shape) <= self.p_not_corrupted)
        ] = -np.inf
        return actual_rss

    def calculate_snr(self, rss: np.ndarray) -> np.ndarray:
        """SNR of specific packet from end device (dim 0) arriving at gateway (dim 1) with considered flat fading and
        interference from other end devices, based on given RSS values previously sampled via `LoraModel.rss`. Packets
        that arrived with an RSS below the gateway sensitivity or that were interfered by a transmission from another
        end device are represented with a value of `-np.inf`.
        """
        return rss - self.thermal_noise

    @cached_property
    def thermal_noise(self) -> float:
        """The intensity of the background thermal noise in dBm. Automatically calculated such that SNR of transmissions
        that just failed is as close as possible to the above values. Reception decision is based purely on RSS and gateway
        sensitivy. This behaviour is implemented to match the implementation in FLoRa as close as possible and permit its use as
        discrete simulation framework."""
        return np.mean(self.GATEWAY_SENSITIVITY - self.REQUIRED_SNR)

    @cached_property
    def p_exceeds_gw_sensitivity(self) -> np.ndarray:
        """Probability of a packet from end device (dim 0) arriving at gateway (dim 1) to exceed the gateway sensitivity.

        - Denoted as &psi;_ik
        """
        # see formula (7)
        return 0.5 + 0.5 * erf(
            # expected RSS of packet from end device arriving at gateway - minimum RSS at gateway for spreading factor of end device
            (
                self.rss_no_flat_fading
                - self.GATEWAY_SENSITIVITY[self._sf - 7][:, np.newaxis]
            )
            / (np.sqrt(2) * self.FLAT_FADING)
        )

    @cached_property
    def transmission_and_symbol_time(self) -> tuple[np.ndarray, np.ndarray]:
        """Time for an end device (dim 0) to transmit a full packet and a single symbol.

        - Denoted as T_i (or T_j) and T_sym^fi
        """
        # time for an end device (dim 0) to transmit a single symbol given its spreading factor
        # see chapter II. PROBLEM FORMULATION, third paragraph
        # denoted as T_sym^fi
        symbol_time = np.power(2, (self._sf)) / (self.BANDWIDTH * 1000)

        # time for an end device (dim 0) to transmit the preamble for starting a new transmission
        # see formula (2)
        # denoted as T_pr^fi
        preamble_time = (self.PREAMBLE_LEN + 4.25) * symbol_time
        # enable low data rate mode for SF11 and SF12
        low_data_rate = np.zeros_like(self._sf)
        low_data_rate[(self._sf >= 11) & (self._sf <= 12)] = 1
        # time for an end device (dim 0) to transmit the payload of a transmission
        # see formula (2)
        # denoted as T_pl^fi
        # NOTE: self.PAYLOAD_SIZE is in bits - formula contains L in bytes (multiplication by 8)
        payload_time = (
            8
            + np.maximum(
                np.ceil(
                    (self.PAYLOAD_SIZE - 4 * self._sf + 28 + 16)
                    / (4 * (self._sf - 2 * low_data_rate))
                )
                * (self.CODING_RATE + 4),
                0,
            )
        ) * symbol_time
        # total time for the transmission of an end device (dim 0)
        # see paragraph between formula (1) and (2)
        # denoted as T_i (or T_j)
        transmission_time = preamble_time + payload_time
        return (transmission_time, symbol_time)

    @cached_property
    def p_not_corrupted(self) -> np.ndarray:
        """Probability of a packet from end device (dim 0) arriving at gateway (dim 1) without being corrupted by
        a simultaneous transmission from any other end device.

        - Denoted as &zeta;_ik
        """
        transmission_time, symbol_time = self.transmission_and_symbol_time

        # time interval during which an end device j (dim 1) would interfer with a transmitting end device i (dim 0)
        # considering the possible overlap (as preamble can be corrupted up to a certain point)
        # see formula (1)
        interference_interval = (
            transmission_time - (self.PREAMBLE_LEN - 5) * symbol_time
        ).reshape(-1, 1) + transmission_time
        # an end device cannot interfere with itself
        np.fill_diagonal(interference_interval, 0)

        # probability of an end device j (dim 1) transmitting at the same time as end device i (dim 0)
        # see formula (4)
        p_interference_transmission = 1 - np.exp(
            (-1) * (1 / self.PACKET_INTERVAL) * interference_interval
        )

        # probability of a simultaneous transmission from an end device j (dim 1) to gateway k (dim 2)
        # being strong enough to interfere with a transmission from end device i (dim 0) to gateway k (dim 2)
        # see formula (9)
        p_interference_rss = 0.5 + 0.5 * erf(
            (
                # SIR threshold for transmitting end device i (dim 0) and interfering end device j (dim 1)
                self.INTER_SF_INTERFERENCE[self._sf - 7, (self._sf - 7).reshape(-1, 1)][
                    :, :, np.newaxis
                ]
                # expected RSS difference between transmissions from end device i (dim 0) and j (dim 1)
                # at gateway k (dim 2)
                - (
                    self.rss_no_flat_fading[:, np.newaxis, :]
                    - self.rss_no_flat_fading[np.newaxis, :, :]
                )
            )
            / (2 * np.sqrt(2) * self.FLAT_FADING)
        )

        # combined probability for no end device j to transmit at the same time as end device i (dim 0) to
        # gateway k (dim 1) while at the same time having a strong enough signal to corrupt
        # the transmission from end device i (dim 0) at gateway k (dim 1)
        return np.prod(
            1 - p_interference_transmission[:, :, np.newaxis] * p_interference_rss,
            axis=1,
        )

    @cached_property
    def pdr_gateway(self) -> np.ndarray:
        """Packet delivery ratio from end device (dim 0) to gateway (dim 1).

        - Denoted as PDR_ik
        """
        # see formula (5)
        return self.p_exceeds_gw_sensitivity * self.p_not_corrupted

    @cached_property
    def pdr(self) -> np.ndarray:
        """Packet delivery ratio from end device (dim 0) to any gateway.

        - Denoted as PDR_i
        """
        # see formula (10)
        return 1 - np.prod(1 - self.pdr_gateway, axis=1)

    @cached_property
    def ee(self) -> np.ndarray:
        """Energy efficiency of an end device (dim 0) as bits successfully transmitted per energy consumed in bits/mJ.

        - Denoted as EE_i
        """
        transmission_time, _ = self.transmission_and_symbol_time
        # see formula (11)
        return self.PAYLOAD_SIZE / (
            np.vectorize(self.transmission_power_consumption)(self._tp)
            * transmission_time
            / np.maximum(self.pdr, self._ALMOST_ZERO)
        )

    @classmethod
    @cache
    def transmission_power_consumption(cls, tp: int) -> float:
        """Actual power consumption for a given LoRa transmission power setting in mW.

        - Denoted as e_p_i
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

    @cached_property
    def max_pdr(self) -> np.ndarray:
        """Estimated maximum achievable packet delivery ratio of an end device (dim 0) based on its distance
        to the closest gateway, not considering any packet interference.
        """
        return np.vectorize(self._calculate_max_pdr)(np.min(self.distance, axis=1))

    @classmethod
    def _calculate_max_pdr(cls, x):
        """The mathematical function to calculate the maximum achievable packet delivery ratio of an end device,
        based on it's distance `x` to the gateway.
        Based on calculation of `pdr` and `rss_no_flat_fading`.
        """
        return 0.5 + 0.5 * erf(
            (
                (
                    cls.ALLOWED_TPS[-1]
                    - cls.PATH_LOSS_REF
                    - 10
                    * cls.PATH_LOSS_EXP
                    * np.log10(np.maximum(x, cls._ALMOST_ZERO) / cls.REF_DIST)
                )
                - cls.GATEWAY_SENSITIVITY[-1]
            )
            / (np.sqrt(2) * cls.FLAT_FADING)
        )
