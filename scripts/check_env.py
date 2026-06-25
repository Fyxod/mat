"""Print the lightweight dependency and GPU checks required before an A6000 run."""
from __future__ import annotations

import importlib
import platform
import sys


def version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as error:
        return f"unavailable ({error})"


def main() -> None:
    print("python executable:", sys.executable)
    print("python version:", sys.version.replace("\n", " "))
    print("platform:", platform.platform())
    try:
        import torch
        print("torch version:", torch.__version__)
        print("torch cuda version:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    except Exception as error:
        print("torch:", f"unavailable ({error})")
    for name in ("diffusers", "transformers", "accelerate", "PIL", "numpy", "pandas", "skimage"):
        print(f"{name} version:", version(name))
    try:
        from diffusers import StableDiffusionInstructPix2PixPipeline  # noqa: F401
        print("InstructPix2Pix import:", "ok")
    except Exception as error:
        print("InstructPix2Pix import:", f"failed ({error})")
    try:
        from transformers import CLIPModel, CLIPProcessor  # noqa: F401
        print("CLIP import:", "ok")
    except Exception as error:
        print("CLIP import:", f"failed ({error})")


if __name__ == "__main__":
    main()
