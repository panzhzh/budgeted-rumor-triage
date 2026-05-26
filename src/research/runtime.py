from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
import subprocess


DATASET_WORKLOAD_HINTS = {
    "PHEME": 6425,
    "CHECKED": 2104,
    "CSDC-Rumor": 324,
}


@dataclass(frozen=True)
class GPUInfo:
    count: int
    device_ids: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class ExecutionPlan:
    dataset_names: tuple[str, ...]
    gpu_info: GPUInfo
    parallel_workers: int
    dataset_assignments: dict[str, str]
    mode: str


def _parse_cuda_visible_devices() -> GPUInfo | None:
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        return None
    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    filtered = tuple(
        part for part in parts if part not in {"-1", "none", "None", "NoDevFiles"}
    )
    return GPUInfo(
        count=len(filtered),
        device_ids=filtered,
        source="CUDA_VISIBLE_DEVICES",
    )


def _detect_with_torch() -> GPUInfo | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        count = int(torch.cuda.device_count())
    except Exception:
        return None
    if count <= 0:
        return GPUInfo(count=0, device_ids=tuple(), source="torch")
    return GPUInfo(
        count=count,
        device_ids=tuple(str(index) for index in range(count)),
        source="torch",
    )


def _detect_with_nvidia_smi() -> GPUInfo | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index",
                "--format=csv,noheader",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    device_ids = tuple(line.strip() for line in output.splitlines() if line.strip())
    return GPUInfo(count=len(device_ids), device_ids=device_ids, source="nvidia-smi")


def detect_visible_gpus() -> GPUInfo:
    for detector in (_parse_cuda_visible_devices, _detect_with_torch, _detect_with_nvidia_smi):
        info = detector()
        if info is not None:
            return info
    return GPUInfo(count=0, device_ids=tuple(), source="none")


def detect_cuda_runtime() -> dict:
    summary = {
        "visible_gpu_count": 0,
        "visible_gpu_source": "none",
        "visible_device_ids": [],
        "torch_cuda_available": False,
        "torch_device_count": 0,
    }
    visible = detect_visible_gpus()
    summary["visible_gpu_count"] = visible.count
    summary["visible_gpu_source"] = visible.source
    summary["visible_device_ids"] = list(visible.device_ids)
    try:
        import torch  # type: ignore

        summary["torch_cuda_available"] = bool(torch.cuda.is_available())
        summary["torch_device_count"] = int(torch.cuda.device_count())
    except Exception as exc:
        summary["torch_error"] = str(exc)
    return summary


def build_execution_plan(
    dataset_names: list[str] | tuple[str, ...],
    max_workers: int | None = None,
) -> ExecutionPlan:
    dataset_tuple = tuple(dataset_names)
    gpu_info = detect_visible_gpus()
    if max_workers is not None and max_workers > 0:
        worker_count = min(max_workers, max(1, len(dataset_tuple)))
    elif gpu_info.count == 0:
        worker_count = 1
    elif gpu_info.count == 1:
        worker_count = min(max(1, len(dataset_tuple)), 3)
    else:
        worker_count = min(len(dataset_tuple), gpu_info.count)

    if gpu_info.count == 0:
        assignments = {name: "cpu" for name in dataset_tuple}
        mode = "cpu_degraded"
    else:
        slots = gpu_info.device_ids or tuple(str(index) for index in range(gpu_info.count))
        assignments = _balanced_dataset_assignments(dataset_tuple, slots)
        mode = "single_gpu_multi_process" if gpu_info.count == 1 else "multi_gpu_parallel"

    return ExecutionPlan(
        dataset_names=dataset_tuple,
        gpu_info=gpu_info,
        parallel_workers=worker_count,
        dataset_assignments=assignments,
        mode=mode,
    )


def _balanced_dataset_assignments(
    dataset_names: tuple[str, ...],
    slots: tuple[str, ...],
) -> dict[str, str]:
    if not slots:
        return {name: "cpu" for name in dataset_names}
    loads = {slot: 0 for slot in slots}
    assignments: dict[str, str] = {}
    ordered = sorted(
        dataset_names,
        key=lambda name: DATASET_WORKLOAD_HINTS.get(name, 1),
        reverse=True,
    )
    for name in ordered:
        slot = min(loads, key=lambda item: (loads[item], item))
        assignments[name] = slot
        loads[slot] += DATASET_WORKLOAD_HINTS.get(name, 1)
    return assignments
