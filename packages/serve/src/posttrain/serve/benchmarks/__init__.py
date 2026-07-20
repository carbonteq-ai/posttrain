"""Serving benchmark workload definitions."""

from .workloads import CORE_INFERENCE_V1, BenchmarkCell, BenchmarkSuite, WorkloadShape

__all__ = ["BenchmarkCell", "BenchmarkSuite", "CORE_INFERENCE_V1", "WorkloadShape"]
