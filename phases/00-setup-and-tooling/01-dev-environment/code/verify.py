import importlib.util
import shutil
import sys


def has_module(name):
    return importlib.util.find_spec(name) is not None


def jupyter_detail():
    for command in ("jupyter", "jupyter-lab", "jupyter-notebook"):
        path = shutil.which(command)
        if path:
            return f"{command} at {path}"
    return "Python package detected"


def torch_module():
    return __import__("torch")


def mps_available():
    torch = torch_module()
    mps = getattr(torch.backends, "mps", None)
    return bool(mps and mps.is_available())


def accelerator_name():
    torch = torch_module()
    if torch.cuda.is_available():
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    if mps_available():
        return "Apple Metal (MPS)"
    return "CPU only"

CHECKS = [
    ("Python 3.10+", lambda: sys.version_info >= (3, 10), f"Python {sys.version}"),
    ("NumPy", lambda: has_module("numpy"), None),
    ("Matplotlib", lambda: has_module("matplotlib"), None),
    ("Jupyter", lambda: has_module("jupyter") or shutil.which("jupyter") is not None, jupyter_detail),
    ("Git", lambda: shutil.which("git") is not None, None),
    ("Node.js", lambda: shutil.which("node") is not None, None),
    ("Rust (cargo)", lambda: shutil.which("cargo") is not None, None),
]

GPU_CHECKS = [
    ("PyTorch", lambda: has_module("torch"), None),
    (
        "CUDA",
        lambda: torch_module().cuda.is_available(),
        lambda: torch_module().cuda.get_device_name(0),
    ),
    ("Apple Metal (MPS)", mps_available, "available"),
    ("Accelerator backend", lambda: has_module("torch"), accelerator_name),
]


def run_check(name, check_fn, detail_fn=None):
    try:
        result = check_fn()
        if result is False:
            raise Exception("Check returned False")
        detail = ""
        if detail_fn:
            if callable(detail_fn):
                detail = f" ({detail_fn()})"
            else:
                detail = f" ({detail_fn})"
        print(f"  [PASS] {name}{detail}")
        return True
    except Exception:
        print(f"  [FAIL] {name}")
        return False


def main():
    print("\n=== AI Engineering from Scratch — Environment Check ===\n")

    print("Core:")
    passed = sum(run_check(name, fn, detail) for name, fn, detail in CHECKS)
    total = len(CHECKS)

    print("\nGPU (optional):")
    gpu_passed = sum(run_check(name, fn, detail) for name, fn, detail in GPU_CHECKS)
    gpu_total = len(GPU_CHECKS)

    print(f"\nResult: {passed}/{total} core checks passed", end="")
    if gpu_passed > 0:
        print(f", {gpu_passed}/{gpu_total} GPU checks passed")
    else:
        print(" (no GPU — that's fine, most lessons work on CPU)")

    if passed == total:
        print("\nYou're ready. Start with Phase 1.\n")
    else:
        print("\nFix the failed checks above, then run this script again.\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
