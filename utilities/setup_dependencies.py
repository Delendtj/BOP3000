"""
Explicit backend dependency setup utility.

Examples:
    python setup_dependencies.py --backend auto
    python setup_dependencies.py --backend openvino
    python setup_dependencies.py --backend tensorrt --dry-run
    python setup_dependencies.py --backend pytorch --skip-common
# Fil litt bygget på hardware.detector
# sjekker om nvidia gpu finnes
# om ja, installerer cuda pakker
# nei = installer openvino
# denne gjør alt automatisk uten brukerinput
# installerer heller ikke pakker som allerede finnes

This script is intentionally separate from normal app startup. It can inspect
the machine, choose a likely backend, and install only the backend-specific
packages you ask for.
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = ROOT / "requirements.txt"

BACKEND_PACKAGES = {
    "tensorrt": [
        ("tensorrt", "tensorrt"),
        ("pycuda", "pycuda"),
    ],
    "openvino": [
        ("openvino", "openvino"),
    ],
    "pytorch": [],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install backend-specific dependencies for this project."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "tensorrt", "openvino", "pytorch"],
        default="auto",
        help="Which backend dependency set to install.",
    )
    parser.add_argument(
        "--skip-common",
        action="store_true",
        help="Skip installing the base project requirements from requirements.txt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be installed without running pip.",
    )
    return parser.parse_args()


def module_exists(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def has_nvidia_smi_gpu() -> tuple[bool, str]:
    """
    Lightweight setup-time heuristic for choosing a backend before runtime.

    This is a heuristic for installation guidance, a
    stricter runtime check is used inside HardwareDetector.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False, "Auto backend selection: 'nvidia-smi' not found."
    except subprocess.TimeoutExpired:
        return False, "Auto backend selection: 'nvidia-smi' timed out."

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode == 0 and names:
        return True, f"Auto backend selection: NVIDIA GPU detected ({', '.join(names)})."

    stderr = result.stderr.strip()
    if stderr:
        return False, f"Auto backend selection: 'nvidia-smi' did not report a usable GPU ({stderr})."
    return False, "Auto backend selection: no NVIDIA GPU reported by 'nvidia-smi'."


def choose_backend(requested: str) -> tuple[str, list[str]]:
    messages: list[str] = []

    if requested != "auto":
        messages.append(f"Backend selection: using explicit backend '{requested}'.")
        return requested, messages

    has_gpu, reason = has_nvidia_smi_gpu()
    messages.append(reason)
    if has_gpu:
        messages.append("Backend selection: choosing 'tensorrt' for setup.")
        return "tensorrt", messages

    messages.append("Backend selection: choosing 'openvino' for setup.")
    return "openvino", messages


def run_pip_install(args: list[str], dry_run: bool) -> bool:
    cmd = [sys.executable, "-m", "pip", "install", "--no-input", *args]
    print("$", " ".join(cmd))
    if dry_run:
        return True

    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        return False


def install_common_requirements(dry_run: bool) -> bool:
    if not REQUIREMENTS_PATH.exists():
        print(f"Base requirements file not found: {REQUIREMENTS_PATH}")
        return False

    print(f"Installing base requirements from {REQUIREMENTS_PATH}")
    return run_pip_install(["-r", str(REQUIREMENTS_PATH)], dry_run=dry_run)


def install_backend_packages(backend: str, dry_run: bool) -> bool:
    ok = True
    packages = BACKEND_PACKAGES.get(backend, [])
    if not packages:
        print(f"No backend-specific packages required for '{backend}'.")
        return True

    for module_name, package_name in packages:
        if module_exists(module_name):
            print(f"Dependency already installed: module '{module_name}'")
            continue

        print(f"Installing backend dependency '{package_name}' for backend '{backend}'")
        if not run_pip_install([package_name], dry_run=dry_run):
            print(f"Failed to install '{package_name}'")
            ok = False
    return ok


def main() -> int:
    args = parse_args()
    backend, messages = choose_backend(args.backend)

    print("Dependency setup")
    for message in messages:
        print("-", message)
    print(f"- Dry run: {args.dry_run}")
    print(f"- Install common requirements: {not args.skip_common}")

    success = True
    if not args.skip_common:
        success = install_common_requirements(dry_run=args.dry_run) and success

    success = install_backend_packages(backend, dry_run=args.dry_run) and success

    if success:
        print("Dependency setup completed successfully.")
        return 0

    print("Dependency setup completed with errors.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
