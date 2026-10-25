"""Optional PyTorch device discovery for lightweight local development."""
from importlib import import_module


DEVICES = ("cpu", "mps", "cuda")


def available_devices() -> tuple[str, ...]:
    available = ["cpu"]
    try:
        torch = import_module("torch")
    except ImportError:
        return tuple(available)
    if getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available():
        available.append("mps")
    if torch.cuda.is_available():
        available.append("cuda")
    return tuple(available)


def resolve_device(requested: str = "auto") -> str:
    """Resolve cpu/mps/cuda/auto without importing torch for a CPU request."""
    if requested not in (*DEVICES, "auto"):
        raise ValueError("device must be one of: auto, cpu, mps, cuda")
    if requested == "cpu":
        return "cpu"
    available = available_devices()
    if requested == "auto":
        return next((device for device in ("cuda", "mps", "cpu") if device in available), "cpu")
    if requested not in available:
        raise RuntimeError(f"Requested device '{requested}' is unavailable; available: {', '.join(available)}")
    return requested
