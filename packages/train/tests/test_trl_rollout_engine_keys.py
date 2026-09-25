"""TRL_ROLLOUT_ENGINE_KEYS lists every rollout-engine key the TRL backend reads.

The performance guard reports binding keys outside this set as ignored, so a
key the backend starts reading must be added here, or the guard would reject a
binding that sets it.
"""

import re
from pathlib import Path

from posttrain.advisor.rules import TRL_ROLLOUT_ENGINE_KEYS

TRL_BACKEND = Path(__file__).parents[1] / "src" / "posttrain" / "train" / "backends" / "trl"
# Engine mappings are read as `engine.get(...)`, `rollout.get(...)` (the
# policy's inference engine) or through a key tuple iterated over the engine.
_READ = re.compile(r"\b(?:engine|rollout)\.get\(\s*\"([a-z_]+)\"")
_ITERATED = re.compile(r"for key in \(([^)]*)\):\s*\n\s*requested = engine\.get\(key\)")


def _keys_read_by_backend() -> set[str]:
    keys: set[str] = set()
    for path in ("common.py", "policy_config.py"):
        source = (TRL_BACKEND / path).read_text(encoding="utf-8")
        keys.update(_READ.findall(source))
        for group in _ITERATED.findall(source):
            keys.update(re.findall(r"\"([a-z_]+)\"", group))
    return keys


def test_every_engine_key_the_trl_backend_reads_is_declared() -> None:
    read = _keys_read_by_backend()
    assert read, "the source scan found no engine reads"
    assert read <= TRL_ROLLOUT_ENGINE_KEYS, sorted(read - TRL_ROLLOUT_ENGINE_KEYS)
