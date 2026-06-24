from __future__ import annotations

import argparse
import json

from phase1.src.runners import run_prompt_discovery
from phase1.src.utils import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean InstructPix2Pix prompt discovery and filtering.")
    parser.add_argument("--root", required=True, help="MAT project root")
    parser.add_argument("--force", action="store_true", help="Recompute completed prompt/settings")
    parser.add_argument("--identity", action="store_true", help="Try optional DeepFace SFace during discovery")
    args = parser.parse_args()
    selected = run_prompt_discovery(project_root(args.root), force=args.force, identity_enabled=args.identity)
    print(json.dumps({"selected_prompts": selected, "count": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
