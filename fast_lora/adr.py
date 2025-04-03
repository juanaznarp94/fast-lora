import numpy as np

from abc import ABC

from .lorawan import Observation, Action
from .configs import CommunicationConfig
from .simulation import LoRaNetwork


class Agent(ABC):
    def add_observation(self, observation: Observation): ...

    def get_action(self) -> Action | None: ...

    def reset(self): ...


class ADR(Agent):
    def __init__(
        self,
        device_margin_db: float,
        num_snr_measurements: int,
        communication_config: CommunicationConfig,
    ):
        self.device_margin_db = device_margin_db
        self.num_snr_measurements = num_snr_measurements
        self._communication_config = communication_config

        self._snr_measurements = []
        self._sf = None

    def add_observation(self, observation):
        # add strongest SNR
        self._snr_measurements.append(max(observation.snr))
        self._sf = observation.sf
        self._tp = observation.tp

    def get_action(self):
        # not enough measurements yet
        if len(self._snr_measurements) < self.num_snr_measurements:
            return None

        # apply ADR algorithm
        self._sf, self._tp = self.adr_algorithm(
            snr_value=np.max(self._snr_measurements),
            required_snr=LoRaNetwork.REQUIRED_SNR,
            device_margin=self.device_margin_db,
            initial_sf=self._sf,
            initial_tp=self._tp,
            allowed_sfs=self._communication_config.allowed_spreading_factors,
            allowed_tps=self._communication_config.allowed_transmission_powers,
        )

        # delete previous measurements
        self._snr_measurements = []

        return Action(sf=self._sf, tp=self._tp)

    def reset(self):
        self._snr_measurements = []
        self._sf = None

    @classmethod
    def adr_algorithm(
        cls,
        snr_value: float,
        required_snr: float,
        device_margin: float,
        initial_sf: int,
        initial_tp: int,
        allowed_sfs: list[int],
        allowed_tps: list[int],
    ):
        # setup
        sf = initial_sf
        tp = initial_tp

        # initial calculations
        snr_margin = snr_value - required_snr - device_margin
        n_step = int(snr_margin / 3)

        # decrease spreading factor as much as possible
        while (n_step > 0) and (sf > min(allowed_sfs)):
            sf = max([s for s in allowed_sfs if s < sf])
            n_step -= 1
        sf = max(sf, min(allowed_sfs))

        # adjust transmission power
        # decrease if possible
        while (n_step > 0) and (tp > min(allowed_tps)):
            tp -= 2
            n_step -= 1
        # increase if necessary
        while (n_step < 0) and (tp < max(allowed_tps)):
            tp += 2
            n_step += 1
        # find closes matching transmission power from allowed values
        for allowed_tp in sorted(allowed_tps):
            if allowed_tp >= tp:
                tp = allowed_tp
                break
        else:
            tp = max(allowed_tp)

        return (sf, tp)
