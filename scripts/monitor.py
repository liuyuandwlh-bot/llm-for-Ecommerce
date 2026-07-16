#!/usr/bin/env python
"""
Monitoring and Metrics Script

Based on recommended plan:
- Prometheus metrics
- Latency tracking
- Cache hit/miss rates
- GPU utilization
"""

import time
import psutil
import json
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    request_id: str
    endpoint: str
    latency_ms: float
    status: str  # success, error
    domain: str
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.requests: list[RequestMetrics] = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors: Dict[str, int] = defaultdict(int)
        self.endpoint_latencies: Dict[str, list] = defaultdict(list)

    def record_request(self, metrics: RequestMetrics):
        """Record a request."""
        self.requests.append(metrics)
        self.endpoint_latencies[metrics.endpoint].append(metrics.latency_ms)

        if metrics.status == "error":
            self.errors[metrics.endpoint] += 1

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses += 1

    def get_summary(self) -> dict:
        """Get metrics summary."""
        total_requests = len(self.requests)

        # Latency percentiles
        all_latencies = [r.latency_ms for r in self.requests]

        latency_summary = {}
        if all_latencies:
            sorted_latencies = sorted(all_latencies)
            latency_summary = {
                "p50": sorted_latencies[len(sorted_latencies) // 2],
                "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
                "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
                "mean": sum(all_latencies) / len(all_latencies),
            }

        # Cache stats
        total_cache = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0

        return {
            "total_requests": total_requests,
            "errors": dict(self.errors),
            "latency_ms": latency_summary,
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
            },
        }


class GPUMonitor:
    """Monitor GPU utilization (requires nvidia-ml-py)."""

    def __init__(self):
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except:
            pass

    def get_metrics(self) -> Optional[dict]:
        """Get current GPU metrics."""
        if not self.available:
            return None

        try:
            import pynvml

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

            return {
                "gpu_utilization_percent": util.gpu,
                "memory_used_mb": mem_info.used // (1024 * 1024),
                "memory_total_mb": mem_info.total // (1024 * 1024),
                "memory_used_percent": mem_info.used / mem_info.total * 100,
            }
        except:
            return None


class SystemMonitor:
    """Monitor system resources."""

    @staticmethod
    def get_metrics() -> dict:
        """Get system metrics."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_used_gb": psutil.virtual_memory().used / (1024 ** 3),
            "memory_total_gb": psutil.virtual_memory().total / (1024 ** 3),
            "memory_percent": psutil.virtual_memory().percent,
        }


def format_prometheus_metrics(metrics: dict, gpu_metrics: dict = None) -> str:
    """Format metrics in Prometheus exposition format."""
    lines = []

    # Request metrics
    lines.append(f"# HELP requests_total Total number of requests")
    lines.append(f"# TYPE requests_total counter")
    lines.append(f"requests_total {metrics.get('total_requests', 0)}")

    lines.append(f"# HELP request_latency_ms Request latency in milliseconds")
    lines.append(f"# TYPE request_latency_ms summary")

    if "latency_ms" in metrics:
        for percentile, value in metrics["latency_ms"].items():
            lines.append(f'request_latency_ms{{quantile="{percentile}"}} {value}')

    # Cache metrics
    cache = metrics.get("cache", {})
    lines.append(f"# HELP cache_hits_total Total cache hits")
    lines.append(f"# TYPE cache_hits_total counter")
    lines.append(f"cache_hits_total {cache.get('hits', 0)}")

    lines.append(f"# HELP cache_hit_rate Cache hit rate")
    lines.append(f"# TYPE cache_hit_rate gauge")
    lines.append(f"cache_hit_rate {cache.get('hit_rate', 0)}")

    # GPU metrics
    if gpu_metrics:
        lines.append(f"# HELP gpu_utilization_percent GPU utilization")
        lines.append(f"# TYPE gpu_utilization_percent gauge")
        lines.append(f'gpu_utilization_percent {{type="gpu"}} {gpu_metrics.get("gpu_utilization_percent", 0)}')

        lines.append(f"# HELP gpu_memory_used_mb GPU memory used")
        lines.append(f"# TYPE gpu_memory_used_mb gauge")
        lines.append(f'gpu_memory_used_mb {{type="gpu"}} {gpu_metrics.get("memory_used_mb", 0)}')

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo
    collector = MetricsCollector()
    gpu_monitor = GPUMonitor()
    sys_monitor = SystemMonitor()

    print("System Metrics:", json.dumps(sys_monitor.get_metrics(), indent=2))
    print("GPU Metrics:", json.dumps(gpu_monitor.get_metrics(), indent=2))
