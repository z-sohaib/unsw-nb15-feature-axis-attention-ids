from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tomllib


REQUIRED_PACKAGES = [
    "torch",
    "pandas",
    "sklearn",
    "imblearn",
    "numpy",
    "matplotlib",
    "pynvml",
    "tqdm",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check thesis experiment setup.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/binary.toml",
            "configs/multiclass.toml",
        ],
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if PyTorch cannot access a CUDA GPU.",
    )
    args = parser.parse_args()

    ok = True
    print("Dependency check")
    for package in REQUIRED_PACKAGES:
        installed = importlib.util.find_spec(package) is not None
        print(f"  {package:<10} {'OK' if installed else 'MISSING'}")
        ok = ok and installed

    gpu_ok = _check_gpu()
    ok = ok and (gpu_ok or not args.require_gpu)

    print("\nDataset path check")
    for config_path in args.configs:
        path = Path(config_path)
        if not path.exists():
            print(f"  {config_path}: CONFIG MISSING")
            ok = False
            continue
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
        data = raw["data"]
        expected_paths = [
            data.get("train_path"),
            data.get("test_path"),
            data.get("normal_path"),
            data.get("attack_path"),
        ]
        print(f"  {config_path}")
        for expected in [p for p in expected_paths if p]:
            exists = Path(expected).exists()
            print(f"    {expected}: {'OK' if exists else 'MISSING'}")
            ok = ok and exists

    if ok:
        print("\nSetup looks ready.")
        return 0
    print("\nSetup is incomplete. Install dependencies and place dataset files as shown above.")
    return 1


def _check_gpu() -> bool:
    print("\nGPU check")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        print("  torch CUDA: UNKNOWN (torch is missing)")
        return False

    import torch

    cuda_available = bool(torch.cuda.is_available())
    print(f"  torch CUDA available: {'YES' if cuda_available else 'NO'}")
    print(f"  torch version: {torch.__version__}")
    print(f"  torch CUDA build: {torch.version.cuda or 'CPU-only'}")
    if cuda_available:
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"  CUDA device {index}: {torch.cuda.get_device_name(index)}")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        print("  nvidia-smi: MISSING")
    else:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print("  nvidia-smi:")
            for line in result.stdout.strip().splitlines():
                print(f"    {line}")
        else:
            print("  nvidia-smi: FOUND but query failed")

    return cuda_available


if __name__ == "__main__":
    raise SystemExit(main())
