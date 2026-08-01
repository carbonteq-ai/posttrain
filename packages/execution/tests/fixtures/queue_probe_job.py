from __future__ import annotations

import argparse
import json
import socket
import time

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()
print(
    json.dumps(
        {
            "event": "started",
            "hostname": socket.gethostname(),
            "label": args.label,
        },
        sort_keys=True,
    ),
    flush=True,
)
time.sleep(args.seconds)
print(
    json.dumps(
        {
            "event": "finished",
            "hostname": socket.gethostname(),
            "label": args.label,
        },
        sort_keys=True,
    ),
    flush=True,
)
