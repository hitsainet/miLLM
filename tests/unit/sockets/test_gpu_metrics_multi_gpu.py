"""GPU metrics describe every card, not the first line of nvidia-smi.

On a two-GPU node the old parser split the whole output on ", " as one line,
int() raised on "36\\n33", and the admin UI showed "No GPU".
"""

import subprocess
from unittest.mock import patch

from millm.sockets.progress import get_gpu_metrics, parse_nvidia_smi_gpus

UUID_3080TI = "GPU-f47ba814-49a2-603f-3595-275284140251"
UUID_3090 = "GPU-247aa582-0d1b-e161-8156-983ed1fefc57"

# index, uuid, utilization, used, total, free, temperature, name
TWO_GPUS = (
    f"0, {UUID_3080TI}, 12, 812, 12288, 11476, 41, NVIDIA GeForce RTX 3080 Ti\n"
    f"1, {UUID_3090}, 88, 20310, 24576, 4266, 67, NVIDIA GeForce RTX 3090\n"
)


def _smi(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=returncode, stdout=stdout, stderr=""
    )


class TestParseNvidiaSmi:
    def test_two_gpus_parse_as_two_entries(self):
        gpus = parse_nvidia_smi_gpus(TWO_GPUS)

        assert [gpu["index"] for gpu in gpus] == [0, 1]
        assert gpus[1] == {
            "index": 1,
            "uuid": UUID_3090,
            "utilization": 88,
            "memory_used_mb": 20310,
            "memory_total_mb": 24576,
            "memory_free_mb": 4266,
            "temperature": 67,
            "name": "NVIDIA GeForce RTX 3090",
        }

    def test_fields_a_card_does_not_expose_count_as_zero(self):
        gpus = parse_nvidia_smi_gpus("0, GPU-x, [N/A], 100, 8192, 8092, [N/A], Tesla T4\n")

        assert gpus == [
            {
                "index": 0,
                "uuid": "GPU-x",
                "utilization": 0,
                "memory_used_mb": 100,
                "memory_total_mb": 8192,
                "memory_free_mb": 8092,
                "temperature": 0,
                "name": "Tesla T4",
            }
        ]

    def test_blank_and_malformed_lines_are_skipped(self):
        stdout = "\n" + TWO_GPUS + "No devices were found\n"

        assert [gpu["index"] for gpu in parse_nvidia_smi_gpus(stdout)] == [0, 1]


class TestGetGpuMetrics:
    def test_two_gpus_report_both_and_aggregate_the_legacy_fields(self):
        with patch("millm.sockets.progress.subprocess.run", return_value=_smi(TWO_GPUS)):
            metrics = get_gpu_metrics()

        assert metrics["gpu_count"] == 2
        assert [gpu["uuid"] for gpu in metrics["gpus"]] == [UUID_3080TI, UUID_3090]
        assert metrics["gpu_memory_total_mb"] == 12288 + 24576
        assert metrics["gpu_memory_used_mb"] == 812 + 20310
        assert metrics["gpu_utilization"] == 50
        assert metrics["gpu_temperature"] == 67

    def test_one_gpu_keeps_its_own_values(self):
        one = f"0, {UUID_3090}, 30, 4000, 24576, 20576, 55, NVIDIA GeForce RTX 3090\n"
        with patch("millm.sockets.progress.subprocess.run", return_value=_smi(one)):
            metrics = get_gpu_metrics()

        assert metrics["gpu_utilization"] == 30
        assert metrics["gpu_memory_used_mb"] == 4000
        assert metrics["gpu_memory_total_mb"] == 24576
        assert metrics["gpu_temperature"] == 55
        assert metrics["gpu_count"] == 1

    def test_name_is_queried_last(self):
        with patch("millm.sockets.progress.subprocess.run", return_value=_smi(TWO_GPUS)) as run:
            get_gpu_metrics()

        query = next(arg for arg in run.call_args.args[0] if arg.startswith("--query-gpu="))
        assert query.endswith(",name"), "a comma in a GPU name would shift the numeric fields"

    def test_missing_nvidia_smi_reports_no_gpus(self):
        with patch("millm.sockets.progress.subprocess.run", side_effect=FileNotFoundError):
            metrics = get_gpu_metrics()

        assert metrics["gpu_count"] == 0
        assert metrics["gpus"] == []
        assert metrics["gpu_memory_total_mb"] == 0
