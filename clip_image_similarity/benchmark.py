from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Iterator, Optional

import torch

from .config import RunConfig
from .serialization import save_json


class BenchmarkRecorder:
    """Collect coarse, cheap benchmark metadata for a successful CLI run."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.started_at = datetime.now(timezone.utc)
        self._start = perf_counter()
        self.timings: dict[str, float] = {}
        self._cuda_enabled = (
            config.device.startswith("cuda") and torch.cuda.is_available()
        )
        self._reset_cuda_peak_memory()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one coarse run stage, synchronizing CUDA work when needed."""
        self._sync_cuda()
        start = perf_counter()
        try:
            yield
        finally:
            self._sync_cuda()
            self.timings[f"{name}_seconds"] = perf_counter() - start

    def save(
        self,
        *,
        image_count: int,
        embedding_dim: Optional[int],
        output_mode: str,
        pairwise_output_path: Path,
    ) -> Path:
        """Write benchmark.json and return its path."""
        self._sync_cuda()
        finished_at = datetime.now(timezone.utc)
        total_seconds = perf_counter() - self._start
        pairwise_output_path = pairwise_output_path.resolve()
        output_bytes = (
            pairwise_output_path.stat().st_size if pairwise_output_path.exists() else 0
        )
        benchmark_path = self.config.output_dir / "benchmark.json"
        save_json(
            {
                "started_at_utc": self.started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "total_seconds": total_seconds,
                **self.timings,
                "model_id": self.config.model_id,
                "pretrained": self.config.pretrained,
                "checkpoint_path": (
                    str(self.config.checkpoint_path)
                    if self.config.checkpoint_path
                    else None
                ),
                "batch_size": self.config.batch_size,
                "device": self.config.device,
                "pairwise_dtype": self.config.pairwise_dtype,
                "top_k": self.config.top_k,
                "image_count": image_count,
                "embedding_dim": embedding_dim,
                "output_mode": output_mode,
                "pairwise_output_path": str(pairwise_output_path),
                "pairwise_output_bytes": output_bytes,
                "images_per_second_total": _rate(image_count, total_seconds),
                "images_per_second_embedding": _rate(
                    image_count, self.timings.get("embedding_seconds")
                ),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                **self._cuda_metadata(),
            },
            benchmark_path,
        )
        return benchmark_path

    def _sync_cuda(self) -> None:
        if self._cuda_enabled:
            torch.cuda.synchronize(self.config.device)

    def _reset_cuda_peak_memory(self) -> None:
        if self._cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.config.device)

    def _cuda_metadata(self) -> dict[str, Optional[int | str]]:
        if not self._cuda_enabled:
            return {
                "cuda_device_name": None,
                "cuda_peak_memory_allocated_bytes": None,
                "cuda_peak_memory_reserved_bytes": None,
            }
        return {
            "cuda_device_name": torch.cuda.get_device_name(self.config.device),
            "cuda_peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                self.config.device
            ),
            "cuda_peak_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                self.config.device
            ),
        }


def _rate(count: int, seconds: Optional[float]) -> Optional[float]:
    if seconds is None or seconds <= 0:
        return None
    return count / seconds
