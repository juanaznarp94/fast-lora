import os
import subprocess
import re
import numpy as np

from typing import Any, Optional

FLORA_ROOT = os.environ.get("FLORA_ROOT")
INET_ROOT = os.environ.get("INET_ROOT")


def run_flora_sim(
    inifile: os.PathLike,
    output_dirname: os.PathLike,
    output_filename: str,
    vector_recording: bool = False,
) -> tuple[os.PathLike, os.PathLike, os.PathLike | None]:
    output_dir = os.path.join(os.getcwd(), output_dirname)
    os.makedirs(output_dir, exist_ok=True)

    logfile = os.path.abspath(os.path.join(output_dir, f"{output_filename}.out"))
    scalar_resultfile = os.path.abspath(
        os.path.join(output_dir, f"{output_filename}.sca")
    )
    vector_resultfile = os.path.abspath(
        os.path.join(output_dir, f"{output_filename}.vec")
    )

    cmd = [
        "opp_run",
        "-l",
        os.path.join(FLORA_ROOT, "src", "flora"),
        "-n",
        f".:{(os.path.join(FLORA_ROOT, 'src'))}:{os.path.join(INET_ROOT, 'src')}",
        "-f",
        os.path.abspath(inifile),
        "-u",
        "Cmdenv",
        "-m",
        "-s",
        "--output-scalar-file",
        scalar_resultfile,
        "--output-vector-file",
        vector_resultfile,
        "--cmdenv-redirect-output",
        "true",
        "--cmdenv-output-file",
        logfile,
        "--**.vector-recording",
        "true" if vector_recording else "false",
    ]
    subprocess.run(
        cmd,
        cwd=os.path.join(FLORA_ROOT, "simulations"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    return (logfile, scalar_resultfile, vector_resultfile if vector_recording else None)


def _extract_values(
    line: str, pattern: re.Pattern, mapping: tuple[callable]
) -> Optional[tuple[Any]]:
    match = re.match(pattern, line)
    if match:
        return (mapping[i](g) for i, g in enumerate(match.groups()))
    else:
        return None


def _get_number_of_nodes(line: str) -> Optional[tuple[int]]:
    pattern = r"config \*\*\.numberOfNodes (\d+)"
    return _extract_values(line, pattern, (int,))


def _get_num_sent_packets(line: str) -> Optional[tuple[int, int]]:
    pattern = r"scalar LoRaNetworkTest\.loRaNodes\[(\d+)\]\.app\[0\] sentPackets (\d+)"
    return _extract_values(line, pattern, (int, int))


def _get_num_received_packets(line: str) -> Optional[tuple[int, int]]:
    pattern = r"scalar LoRaNetworkTest\.networkServer\.app\[0\] \"numReceivedFromNode (\d+)\" (\d+)"
    return _extract_values(line, pattern, (int, int))


def _get_consumed_energy(line: str) -> Optional[tuple[int, float]]:
    pattern = r"scalar LoRaNetworkTest\.loRaNodes\[(\d+)\]\.LoRaNic\.radio\.energyConsumer totalEnergyConsumed (\d+\.\d+)"
    return _extract_values(line, pattern, (int, float))


def _get_payload_size(line: str) -> Optional[tuple[int, int]]:
    pattern = r"par LoRaNetworkTest\.loRaNodes\[(\d+)\]\.app\[0\] dataSize (\d+)B"
    return _extract_values(line, pattern, (int, int))


def get_metrics(scalarfile: os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    # read scalarfile containing simulation results
    with open(scalarfile, "r") as file:
        for line in file:
            if _get_number_of_nodes(line):
                (number_of_nodes,) = _get_number_of_nodes(line)
                # setup matrices for other data
                consumed_energy = np.zeros((number_of_nodes,), dtype=np.float64)
                num_sent_packets = np.zeros((number_of_nodes,), dtype=np.int32)
                num_received_packets = np.zeros((number_of_nodes,), dtype=np.int32)
                payload_size = np.zeros((number_of_nodes,), dtype=np.int32)

            # extract simulation results
            elif _get_consumed_energy(line):
                node_id, val = _get_consumed_energy(line)
                consumed_energy[node_id] = val
            elif _get_num_sent_packets(line):
                node_id, val = _get_num_sent_packets(line)
                num_sent_packets[node_id] = val
            elif _get_num_received_packets(line):
                node_id, val = _get_num_received_packets(line)
                num_received_packets[node_id] = val
            elif _get_payload_size(line):
                node_id, val = _get_payload_size(line)
                payload_size[node_id] = val

    # calculate desired metrics
    pdr = num_received_packets / num_sent_packets
    ee = (num_received_packets * payload_size * 8) / (consumed_energy * 1000)

    return (pdr, ee)
