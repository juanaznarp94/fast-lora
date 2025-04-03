import numpy as np
import pandas as pd

from .simulation import LoRaNetwork


class Observation:
    def __init__(
        self, rss: list[float], snr: list[float], sf: int, tp: int, timestamp: float
    ):
        self.rss = rss.copy()
        self.snr = snr.copy()
        self.sf = sf
        self.tp = tp
        self.timestamp = timestamp


class Action:
    def __init__(self, sf: int, tp: int):
        self.sf = sf
        self.tp = tp


class Metrics:
    def __init__(
        self,
        timestamp: float,
        pos: np.ndarray,
        pdr: np.ndarray,
        ee: np.ndarray,
        sf: np.ndarray,
        tp: np.ndarray,
    ):
        self.timestamp = timestamp
        self.pos = pos.copy()
        self.pdr = pdr.copy()
        self.ee = ee.copy()
        self.sf = sf.copy()
        self.tp = tp.copy()

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": self.timestamp,
                "end_device_id": np.arange(self.pos.shape[0]),
                "pos_x": self.pos[:, 0],
                "pos_y": self.pos[:, 1],
                "pdr": self.pdr,
                "ee": self.ee,
                "sf": self.sf,
                "tp": self.tp,
            }
        )


class LoRaWANNetwork:
    def __init__(
        self,
        lora_network: LoRaNetwork,
        blind_adr: np.ndarray | bool,
        adr_ack_limit: np.ndarray | int,
        adr_ack_delay: np.ndarray | int,
        max_simulation_time_seconds: int = None,
        seed: int = 0,
    ):
        self.lora_network = lora_network
        num_end_devices = self.lora_network.end_devices.positions.shape[0]

        self.blind_adr = (
            np.full((num_end_devices,), blind_adr)
            if isinstance(blind_adr, bool)
            else blind_adr
        )
        self.adr_ack_limit = (
            np.full((num_end_devices,), adr_ack_limit)
            if isinstance(adr_ack_limit, int)
            else adr_ack_limit
        )
        self.adr_ack_delay = (
            np.full((num_end_devices,), adr_ack_delay)
            if isinstance(adr_ack_delay, int)
            else adr_ack_delay
        )

        # time configuration
        self._max_simulation_time_seconds = max_simulation_time_seconds
        self._simulation_timestep = (
            self.lora_network.communication_config.packet_interval
        )

        # state management
        self._simulation_time_seconds = 0
        self._adr_ack_count = np.full(
            (self.lora_network.end_devices.positions.shape[0],), 0
        )

    def get_obversations(self) -> dict[int, Observation]:
        # increase simulation time
        self._simulation_time_seconds += self._simulation_timestep

        # collect data
        rss = self.lora_network.rss_sampled
        snr = self.lora_network.calculate_snr(rss)
        sf = self.lora_network.end_devices.spreading_factors
        tp = self.lora_network.end_devices.transmission_powers

        # generate individual observations
        message_received = np.max(rss, axis=1) > -np.inf
        adr_requested = self._adr_ack_count >= self.adr_ack_limit
        blind_adr = self.blind_adr
        agent_ids = np.nonzero(message_received | blind_adr)[0]

        # increase ADR ACK counter for those end devices where no uplink message arrived at any gateway
        # reset ADR ACK counter for those end device where an uplink message arrived at any gateway
        # if ADR ACK counter indicates a request
        # assume that downlink message containing SF/TP info can be received whenever uplink transmission has been successful
        self._adr_ack_count[(message_received & adr_requested) | blind_adr] = 0
        self._adr_ack_count[~(message_received & adr_requested) | blind_adr] += 1

        return {
            agent_id: Observation(
                timestamp=self._simulation_time_seconds,
                rss=rss[agent_id].tolist(),
                snr=snr[agent_id].tolist(),
                sf=sf[agent_id],
                tp=tp[agent_id],
            )
            for agent_id in agent_ids
        }

    def apply_actions(self, actions: dict[int, Action]):
        # apply agent controlled actions
        sf = np.full((self.lora_network.end_devices.positions.shape[0],), np.nan)
        tp = np.full((self.lora_network.end_devices.positions.shape[0],), np.nan)
        for agent_id, action in actions.items():
            sf[agent_id] = action.sf
            tp[agent_id] = action.tp
            # remember that update has been sent
            self._adr_ack_count[agent_id] = 0

        # apply recovery mechanism as described in: https://learn.semtech.com/mod/book/view.php?id=174&chapterid=162
        # initially increase TP to allowed maximum
        recovery_mask_tp = (
            self._adr_ack_count == self.adr_ack_limit + self.adr_ack_delay
        )
        if recovery_mask_tp.any():
            tp = np.where(
                recovery_mask_tp,
                np.max(
                    self.lora_network.communication_config.allowed_transmission_powers
                ),
                tp,
            )
        # thereafter start increasing SF up to allowed maximum
        recovery_mask_sf = (
            (self._adr_ack_count - self.adr_ack_limit) / self.adr_ack_delay == 0
        ) & (self._adr_ack_count - self.adr_ack_limit - self.adr_ack_delay > 0)
        if recovery_mask_sf.any():
            sf = np.where(
                recovery_mask_sf,
                np.minimum(
                    self.lora_network.end_devices.spreading_factors + 1,
                    np.max(
                        self.lora_network.communication_config.allowed_spreading_factors
                    ),
                ),
                sf,
            )

        # apply changes to network
        self.lora_network.end_devices.spreading_factors = sf
        self.lora_network.end_devices.transmission_powers = tp

    def get_metrics(self) -> Metrics:
        return Metrics(
            timestamp=float(self._simulation_time_seconds),
            pos=self.lora_network.end_devices.positions,
            pdr=self.lora_network.pdr,
            ee=self.lora_network.ee,
            sf=self.lora_network.end_devices.spreading_factors,
            tp=self.lora_network.end_devices.transmission_powers,
        )

    @property
    def terminated(self) -> bool:
        return (
            self._simulation_time_seconds >= self._max_simulation_time_seconds
            if self._max_simulation_time_seconds is not None
            else False
        )
