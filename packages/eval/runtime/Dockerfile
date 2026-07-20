FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Verifiers prepares harness scripts with `pip install -U --user uv`. Installing
# the same pinned uv build through pip makes that check a fast no-op while the
# upstream image still supplies the native binary.
RUN python -m pip install --no-cache-dir --user "uv==0.11.29"

# The runtime setup always asks pip to upgrade uv. Prevent that redundant network
# lookup; uv itself remains online for PEP 723 harness dependencies.
ENV PIP_NO_INDEX=1

# Warm the exact content-addressed script environment used by the null harness in
# Verifiers commit 284a868d. The runtime overwrites this dependency-equivalent
# stub with the full pinned program before execution.
COPY null_harness_warmup.py /tmp/vf-scripts/a849337f8e4e87e3e5a2da49c420f69eb5837f539c7fff9be7cbd37abc6b373a.py
RUN uv sync --script /tmp/vf-scripts/a849337f8e4e87e3e5a2da49c420f69eb5837f539c7fff9be7cbd37abc6b373a.py --no-config

# The pinned harness dependency set is now present in both the script environment
# and uv cache. Avoid nondeterministic index refreshes in per-rollout sandboxes.
ENV UV_OFFLINE=1
