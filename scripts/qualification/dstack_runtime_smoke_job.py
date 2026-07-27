from __future__ import annotations

import argparse
import importlib
import json
import socket

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=("train", "serve"), default="train")
args = parser.parse_args()

payload = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "hostname": socket.gethostname(),
    "runtime_profile": args.profile,
    "torch": torch.__version__,
}
if args.profile == "serve":
    vllm = importlib.import_module("vllm")
    serve = importlib.import_module("posttrain.serve")
    assert callable(serve.benchmark)
    payload["vllm"] = vllm.__version__
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"]:
    raise SystemExit("CUDA is unavailable")
