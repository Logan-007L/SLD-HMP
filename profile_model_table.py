#!/usr/bin/env python3
import argparse
import csv
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from motion_pred.utils.config import Config
from models.motion_pred import get_model


@dataclass
class DummyDataset:
    kept_joints: np.ndarray
    traj_dim: int


def build_dummy_dataset(dataset_name: str) -> DummyDataset:
    dataset_name = dataset_name.lower()
    if dataset_name == "h36m":
        removed_joints = {4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31}
        kept_joints = np.array([x for x in range(32) if x not in removed_joints], dtype=np.int64)
    elif dataset_name == "humaneva":
        kept_joints = np.arange(15, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Only h36m/humaneva are supported.")
    traj_dim = (kept_joints.shape[0] - 1) * 3
    return DummyDataset(kept_joints=kept_joints, traj_dim=traj_dim)


def parse_int(value: str, default_value: int) -> int:
    if value is None:
        return default_value
    value = str(value).strip()
    if value == "":
        return default_value
    return int(float(value))


def resolve_device(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        print(f"[Warn] CUDA is unavailable, fallback to CPU (requested={device_text}).")
        return torch.device("cpu")
    return device


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def create_sample_input(
    cfg: Config,
    traj_dim: int,
    batch_size: int,
    input_mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    if input_mode == "history":
        t_in = cfg.t_his
    elif input_mode == "full":
        t_in = cfg.t_his + cfg.t_pred
    else:
        raise ValueError(f"Unsupported input_mode '{input_mode}', must be history/full.")

    sample = torch.randn(t_in, batch_size, traj_dim, dtype=torch.float32, device=device)
    return sample, t_in


def measure_flops(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> Tuple[float, str]:
    try:
        from thop import profile as thop_profile  # type: ignore

        flops, _ = thop_profile(model, inputs=(sample_input,), verbose=False)
        if flops and flops > 0:
            return float(flops), "thop"
    except Exception:
        pass

    try:
        activities: List[torch.profiler.ProfilerActivity] = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.no_grad():
            with torch.profiler.profile(activities=activities, with_flops=True) as prof:
                _ = model(sample_input)
                synchronize_if_needed(device)

        flops = 0.0
        for item in prof.key_averages():
            item_flops = getattr(item, "flops", 0)
            if item_flops:
                flops += float(item_flops)
        if flops > 0:
            return flops, "torch.profiler"
    except Exception:
        pass

    return math.nan, "unavailable"


@torch.no_grad()
def measure_inference_time_ms(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
) -> float:
    model.eval()
    for _ in range(max(0, warmup)):
        _ = model(sample_input)
    synchronize_if_needed(device)

    t0 = time.perf_counter()
    for _ in range(max(1, repeat)):
        _ = model(sample_input)
    synchronize_if_needed(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / max(1, repeat)


def read_csv_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_one_row(
    row: Dict[str, str],
    default_cfg: str,
    default_batch_size: int,
    default_device: str,
    default_input_mode: str,
    default_warmup: int,
    default_repeat: int,
) -> Dict[str, str]:
    out = dict(row)

    cfg_id = (row.get("cfg") or default_cfg or "").strip()
    if not cfg_id:
        raise ValueError("Each row must provide 'cfg' column or pass --cfg.")

    batch_size = parse_int(row.get("batch_size"), default_batch_size)
    warmup = parse_int(row.get("warmup"), default_warmup)
    repeat = parse_int(row.get("repeat"), default_repeat)
    input_mode = (row.get("input_mode") or default_input_mode).strip().lower()
    device_text = (row.get("device") or default_device).strip().lower()
    device = resolve_device(device_text)

    cfg = Config(cfg_id, test=True)
    dummy_dataset = build_dummy_dataset(cfg.dataset)

    model, _ = get_model(cfg, dummy_dataset, cfg.dataset)
    model = model.to(device).eval()

    sample_input, t_in = create_sample_input(
        cfg=cfg,
        traj_dim=dummy_dataset.traj_dim,
        batch_size=batch_size,
        input_mode=input_mode,
        device=device,
    )

    total_params, trainable_params = count_params(model)
    flops, flops_backend = measure_flops(model, sample_input, device)
    inference_time_ms = measure_inference_time_ms(
        model=model,
        sample_input=sample_input,
        device=device,
        warmup=warmup,
        repeat=repeat,
    )

    out["cfg"] = cfg_id
    out["dataset"] = cfg.dataset
    out["device"] = str(device)
    out["input_mode"] = input_mode
    out["batch_size"] = str(batch_size)
    out["warmup"] = str(warmup)
    out["repeat"] = str(repeat)
    out["t_input"] = str(t_in)
    out["params_total"] = str(total_params)
    out["params_trainable"] = str(trainable_params)
    out["params_million"] = f"{total_params / 1e6:.6f}"
    out["flops"] = "" if math.isnan(flops) else f"{flops:.0f}"
    out["flops_g"] = "" if math.isnan(flops) else f"{flops / 1e9:.6f}"
    out["flops_backend"] = flops_backend
    out["inference_time_ms"] = f"{inference_time_ms:.6f}"
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a model table CSV file and append Params/FLOPs/Inference Time columns. "
            "Input CSV must contain at least one row and either a 'cfg' column or --cfg."
        )
    )
    parser.add_argument("--input-table", required=True, help="Input CSV table path.")
    parser.add_argument("--output-table", required=True, help="Output CSV table path.")
    parser.add_argument("--cfg", default="", help="Default cfg ID if a row does not set cfg.")
    parser.add_argument("--batch-size", type=int, default=1, help="Default batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Default device: auto/cpu/cuda/cuda:0 ...",
    )
    parser.add_argument(
        "--input-mode",
        default="history",
        choices=["history", "full"],
        help="history uses t_his input, full uses t_his+t_pred input.",
    )
    parser.add_argument("--warmup", type=int, default=20, help="Default warmup iterations.")
    parser.add_argument("--repeat", type=int, default=50, help="Default timed iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_fields, rows = read_csv_rows(args.input_table)
    if len(rows) == 0:
        raise ValueError("Input CSV contains no data rows.")

    output_rows = []
    total_rows = len(rows)
    for idx, row in enumerate(rows, start=1):
        cfg_value = (row.get("cfg") or args.cfg or "").strip()
        print(f"[{idx}/{total_rows}] Profiling cfg={cfg_value if cfg_value else '<empty>'} ...")
        profiled_row = process_one_row(
            row=row,
            default_cfg=args.cfg,
            default_batch_size=args.batch_size,
            default_device=args.device,
            default_input_mode=args.input_mode,
            default_warmup=args.warmup,
            default_repeat=args.repeat,
        )
        output_rows.append(profiled_row)

    extra_columns = [
        "dataset",
        "device",
        "input_mode",
        "batch_size",
        "warmup",
        "repeat",
        "t_input",
        "params_total",
        "params_trainable",
        "params_million",
        "flops",
        "flops_g",
        "flops_backend",
        "inference_time_ms",
    ]
    output_fields = list(input_fields)
    for col in extra_columns:
        if col not in output_fields:
            output_fields.append(col)

    write_csv_rows(args.output_table, output_fields, output_rows)
    print(f"Done. Output written to: {args.output_table}")


if __name__ == "__main__":
    main()
