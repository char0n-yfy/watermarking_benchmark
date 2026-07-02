from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import system_metrics as metrics


class SystemMetricsWindowsTest(unittest.TestCase):
    def test_windows_memory_metrics_use_psutil(self) -> None:
        fake_memory = type("VM", (), {"total": 16_000, "used": 8_000, "available": 8_000})()
        with patch.object(metrics, "psutil") as psutil_mock, patch.object(metrics, "_is_windows", return_value=True):
            psutil_mock.virtual_memory.return_value = fake_memory
            result = metrics._windows_memory_metrics()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["totalBytes"], 16_000)
        self.assertEqual(result["usedPercent"], 50.0)

    def test_windows_cpu_metrics_use_psutil(self) -> None:
        with patch.object(metrics, "psutil") as psutil_mock, patch.object(metrics, "_is_windows", return_value=True):
            psutil_mock.cpu_percent.return_value = 42.5
            result = metrics._windows_cpu_usage_percent()
        self.assertEqual(result, 42.5)

    def test_collect_system_metrics_on_windows(self) -> None:
        fake_memory = type("VM", (), {"total": 32_000, "used": 16_000, "available": 16_000})()
        fake_process = type("Proc", (), {"memory_info": lambda self: type("MI", (), {"rss": 123456})()})()
        fake_counters = type("Counters", (), {"read_bytes": 1000, "write_bytes": 2000})()
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(metrics, "platform") as platform_mock, patch.object(metrics, "psutil") as psutil_mock, patch.object(
                metrics, "_gpu_metrics", return_value={"available": False, "devices": []}
            ), patch.object(metrics, "_run_command", return_value=""):
                platform_mock.system.return_value = "Windows"
                platform_mock.platform.return_value = "Windows-11"
                platform_mock.machine.return_value = "AMD64"
                platform_mock.python_version.return_value = "3.12.0"
                psutil_mock.virtual_memory.return_value = fake_memory
                psutil_mock.cpu_percent.return_value = 12.3
                psutil_mock.boot_time.return_value = metrics.time.time() - 3600
                psutil_mock.Process.return_value = fake_process
                psutil_mock.disk_io_counters.return_value = fake_counters
                payload = metrics.collect_system_metrics(data_root=Path(tmp), device="cpu")

        self.assertEqual(payload["memory"]["usedPercent"], 50.0)
        self.assertEqual(payload["cpu"]["usagePercent"], 12.3)
        self.assertEqual(payload["process"]["rssBytes"], 123456)
        self.assertIsNotNone(payload["uptimeSeconds"])
        self.assertIn("temperatureC", payload["cpu"])
        self.assertIn("powerDrawW", payload["cpu"])

    def test_windows_cpu_temperature_uses_lhm_http(self) -> None:
        payload = {
            "Children": [
                {
                    "Type": "Temperature",
                    "Text": "CPU Package",
                    "Value": "72.5 °C",
                    "Children": [],
                }
            ]
        }
        with patch.object(metrics, "_fetch_lhm_http_payload", return_value=payload):
            self.assertEqual(metrics._windows_lhm_http_cpu_temperature_c(), 72.5)
            self.assertEqual(metrics._windows_cpu_temperature_c(), 72.5)

    def test_windows_cpu_temperature_uses_perf_counter(self) -> None:
        with patch.object(metrics, "_is_windows", return_value=True), patch.object(
            metrics, "_windows_lhm_http_cpu_temperature_c", return_value=None
        ), patch.object(
            metrics,
            "_run_powershell_counter",
            side_effect=[3010.0, None],
        ):
            self.assertEqual(metrics._windows_cpu_temperature_c(), 27.9)

    def test_windows_cpu_temperature_parses_wmi_value(self) -> None:
        with patch.object(metrics, "_is_windows", return_value=True), patch.object(
            metrics, "_windows_lhm_http_cpu_temperature_c", return_value=None
        ), patch.object(metrics, "_run_powershell_counter", return_value=None), patch.object(
            metrics,
            "_run_command",
            return_value="3000\n",
        ):
            self.assertEqual(metrics._windows_cpu_temperature_c(), 26.9)

    def test_windows_cpu_power_uses_perf_counter(self) -> None:
        with patch.object(metrics, "_is_windows", return_value=True), patch.object(
            metrics, "_windows_lhm_http_cpu_power_w", return_value=None
        ), patch.object(metrics, "_windows_hardware_monitor_cpu_power_w", return_value=None), patch.object(
            metrics, "_windows_pdh_cpu_power_w", return_value=None
        ), patch.object(metrics, "_windows_powershell_cpu_power_w", return_value=16.0):
            self.assertEqual(metrics._windows_cpu_power_draw_w(), 16.0)

    def test_lhm_http_power_parses_cpu_package(self) -> None:
        payload = {
            "Children": [
                {
                    "Type": "Power",
                    "Text": "CPU Package",
                    "RawValue": "17.2 W",
                    "Children": [],
                }
            ]
        }
        preferred = tuple(name.lower() for name in metrics._LHM_POWER_SENSOR_NAMES)
        self.assertEqual(metrics._lhm_power_from_json_tree(payload, preferred), 17.2)

    def test_windows_lhm_http_cpu_power_w(self) -> None:
        payload = {
            "Children": [{"Type": "Power", "Text": "CPU Package", "Value": "19.7 W", "Children": []}]
        }
        metrics._LHM_POWER_RAW_SAMPLES = []
        with patch.object(metrics, "_fetch_lhm_http_payload", return_value=payload):
            self.assertEqual(metrics._windows_lhm_http_cpu_power_w(), 19.7)

    def test_lhm_http_client_disables_proxy(self) -> None:
        metrics._LHM_HTTP_CLIENT = None
        with patch.object(metrics, "httpx") as httpx_mock:
            metrics._get_lhm_http_client()
        httpx_mock.Client.assert_called_once_with(trust_env=False, timeout=2.0)
        metrics._LHM_HTTP_CLIENT = None

    def test_windows_powershell_cpu_power_converts_milliwatts(self) -> None:
        with patch.object(metrics, "_run_powershell_counter", return_value=16009.0):
            self.assertEqual(metrics._windows_powershell_cpu_power_w(), 16.0)

    def test_windows_pdh_cpu_power_converts_milliwatts(self) -> None:
        metrics._reset_windows_pdh_power()
        fake_query = object()
        fake_counter = object()
        with patch.object(metrics, "win32pdh") as pdh_mock, patch.object(metrics.time, "sleep"):
            pdh_mock.OpenQuery.return_value = fake_query
            pdh_mock.AddCounter.return_value = fake_counter
            pdh_mock.GetFormattedCounterValue.return_value = (0, 13580.0)
            self.assertEqual(metrics._windows_pdh_cpu_power_w(), 13.6)
        metrics._reset_windows_pdh_power()

    def test_cpu_power_smoothing_keeps_last_value(self) -> None:
        metrics._CPU_POWER_W_EMA = 12.0
        self.assertEqual(metrics._smooth_cpu_power_w(None), 12.0)
        self.assertEqual(metrics._smooth_cpu_power_w(20.0), 15.6)
        metrics._CPU_POWER_W_EMA = None

    def test_median_lhm_power_w(self) -> None:
        metrics._LHM_POWER_RAW_SAMPLES = []
        self.assertEqual(metrics._median_lhm_power_w(30.0), 30.0)
        self.assertEqual(metrics._median_lhm_power_w(40.0), 40.0)
        self.assertEqual(metrics._median_lhm_power_w(26.0), 30.0)
        metrics._LHM_POWER_RAW_SAMPLES = []

    def test_warmup_cpu_power_sensors_stops_after_first_reading(self) -> None:
        metrics._CPU_POWER_W_EMA = None
        with patch.object(metrics, "_is_windows", return_value=True), patch.object(
            metrics, "_windows_cpu_power_draw_w", side_effect=[None, 18.5]
        ), patch.object(metrics.time, "sleep"):
            metrics.warmup_cpu_power_sensors(attempts=3, delay_seconds=0.1)
        self.assertEqual(metrics._CPU_POWER_W_EMA, 18.5)

    def test_linux_cpu_power_uses_rapl_delta(self) -> None:
        metrics._RAPL_ENERGY_SAMPLE = None
        with patch.object(metrics, "_read_rapl_energy_microjoules", side_effect=[1_000_000, 11_000_000]), patch.object(
            metrics.time, "monotonic", side_effect=[0.0, 1.0]
        ):
            self.assertIsNone(metrics._linux_cpu_power_draw_w())
            self.assertEqual(metrics._linux_cpu_power_draw_w(), 10.0)

    def test_nvidia_smi_windows_path(self) -> None:
        with patch.object(metrics, "_is_windows", return_value=True), patch.dict(
            metrics.os.environ,
            {"ProgramFiles": r"C:\Program Files"},
            clear=False,
        ), patch.object(metrics.Path, "is_file", return_value=True):
            executable = metrics._nvidia_smi_executable()
        self.assertTrue(executable.endswith("nvidia-smi.exe"))


if __name__ == "__main__":
    unittest.main()
