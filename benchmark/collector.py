# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import requests
import yaml

logging = logging.getLogger("nn-Meter")

class PrometheusCollector:
    def __init__(self, config_path="nn_meter/hase/config.yaml"):
        self.url = "http://localhost:9090"
        # internal keys: sm_active / sm_occupancy / dram_active
        self.metrics_map = {
            "sm_active": "DCGM_FI_PROF_SM_ACTIVE",
            "sm_occupancy": "DCGM_FI_PROF_SM_OCCUPANCY",
            "dram_active": "DCGM_FI_PROF_DRAM_ACTIVE",
        }
        
        # Load config if available
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                    if "PROMETHEUS" in cfg:
                        self.url = cfg["PROMETHEUS"].get("URL", self.url)
                        metrics_cfg = cfg["PROMETHEUS"].get("METRICS", {}) or {}
                        # Support config keys in either UPPER_CASE or internal snake_case
                        for k, v in metrics_cfg.items():
                            if not v:
                                continue
                            key = str(k).strip()
                            metric = str(v).strip()
                            if key in ("SM_ACTIVE", "sm_active"):
                                self.metrics_map["sm_active"] = metric
                            elif key in ("SM_OCCUPANCY", "sm_occupancy"):
                                self.metrics_map["sm_occupancy"] = metric
                            elif key in ("DRAM_ACTIVE", "dram_active"):
                                self.metrics_map["dram_active"] = metric
        except Exception as e:
            logging.warning(f"Failed to load Prometheus config: {e}. Using defaults.")

    @staticmethod
    def _window_seconds_to_prom_duration(window_seconds: float) -> str:
        """
        Convert seconds to a PromQL duration string.
        We intentionally keep it in seconds for correctness (e.g., '90s').
        """
        try:
            sec = int(round(float(window_seconds)))
        except Exception:
            sec = 1
        sec = max(1, sec)
        return f"{sec}s"

    def query_instant(self, promql: str, ts: float | None = None) -> float | None:
        """
        Query Prometheus instant API and return the first scalar-like value.
        The averaging/aggregation should be done by Prometheus (PromQL) side.
        """
        try:
            query_url = f"{self.url}/api/v1/query"
            params: dict = {"query": promql}
            if ts is not None:
                params["time"] = ts

            response = requests.get(query_url, params=params)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "success":
                return None

            result = (data.get("data") or {}).get("result") or []
            if not result:
                return None

            # result[0]["value"] is [timestamp, "value"]
            value = result[0].get("value")
            if not value or len(value) < 2:
                return None

            v = float(value[1])
            if v != v:  # NaN
                return None
            return v
        except Exception as e:
            logging.error(f"Prometheus instant query failed: promql={promql}, err={e}")
            return None

    def get_averages(self, start_time, end_time):
        """
        Get average values for all configured metrics during the window.
        """
        window = self._window_seconds_to_prom_duration(float(end_time) - float(start_time))

        # Use Prometheus to compute averages: avg_over_time over the time window ending at end_time.
        # We use subquery form so metric_name can be a selector or an instant-vector expression.
        stats: dict = {}
        for key, metric_name in self.metrics_map.items():
            promql = f"avg_over_time(({metric_name})[{window}:])"
            v = self.query_instant(promql, ts=end_time)
            stats[key] = float(v) if v is not None else 0.0

        # Derived metric: occupancy_when_active
        # Many PROF metrics are effectively time-window averages; when SM_ACTIVE is low,
        # raw SM_OCCUPANCY will also look low even if "occupancy during active time" is high.
        # This derived value estimates occupancy conditioned on being active.
        sm_active = float(stats.get("sm_active", 0.0) or 0.0)
        sm_occ = float(stats.get("sm_occupancy", 0.0) or 0.0)

        # Heuristic for units:
        # - DCGM often reports percentages in [0, 100]
        # - If values appear in [0, 1], treat them as ratios
        if sm_active > 1.5:
            denom = max(sm_active / 100.0, 1e-6)
            stats["sm_occupancy_when_active"] = sm_occ / denom
        else:
            denom = max(sm_active, 1e-6)
            stats["sm_occupancy_when_active"] = sm_occ / denom

        return stats
