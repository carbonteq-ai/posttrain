#!/usr/bin/env bash
# Submit one lean SDPO A/B phase as a dstack task on local-gpu-workers (plan D17, D20).
# Usage: sdpo_ab_launch.sh <run-folder> <max-duration> <python args...>
# Refuses if any dstack run is active or any GPU compute process exists.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
envrepo="${SDPO_AB_ENV_REPO:?set SDPO_AB_ENV_REPO to a verifiers-environments checkout}"
envcommit="${SDPO_AB_ENV_COMMIT:?set SDPO_AB_ENV_COMMIT to the environment commit (no default)}"  # D25: point at the repaired environment commit
dstack="${DSTACK_BIN:-dstack}"
scratch=${SDPO_AB_STAGE:?set SDPO_AB_STAGE to a staging directory}
ws=${SDPO_AB_WS:?set SDPO_AB_WS to the read-only ssh wrapper}
folder="$1"; duration="$2"; shift 2
[[ "$folder" =~ ^sdpo-ab-[a-z0-9-]+$ ]] || { echo "bad run folder" >&2; exit 1; }
(( ${#folder} <= 41 )) || { echo "run folder longer than 41 characters: $folder" >&2; exit 1; }
[[ -z "$(git -C "$repo" status --porcelain scripts/qualification)" ]] || { echo "commit code first" >&2; exit 1; }
# every shipped input must be a file committed in this repository, unchanged since the commit
for extra in ${SDPO_AB_INPUTS:-}; do
  [[ -f "$extra" ]] && git -C "$repo" ls-files --error-unmatch -- "$(realpath "$extra")" >/dev/null 2>&1 \
    && [[ -z "$(git -C "$repo" status --porcelain -- "$(realpath "$extra")")" ]] \
    || { echo "input not committed in $repo: $extra" >&2; exit 1; }
done
sha="$(git -C "$repo" rev-parse HEAD)"
stage="$scratch/$folder"
[[ ! -e "$stage" ]] || { echo "stage exists: $stage" >&2; exit 1; }
mkdir -p "$stage"
git -C "$repo" archive --format=tar.gz -o "$stage/code.tgz" HEAD scripts/qualification
git -C "$envrepo" archive --format=tar.gz -o "$stage/env.tgz" "$envcommit" environments/automationbench_v1/src
# dstack limits each uploaded archive to 2 MiB; ship the environment in parts.
split -b 1800k -d "$stage/env.tgz" "$stage/env.tgz.part"
envparts="$(cd "$stage" && ls env.tgz.part*)"
git -C "$envrepo" rev-parse --verify "$envcommit^{commit}" > "$stage/env-commit.txt"
for extra in ${SDPO_AB_INPUTS:-}; do cp "$extra" "$stage/"; done
( cd "$stage" && sha256sum ./*.tgz ./env-commit.txt $(for e in ${SDPO_AB_INPUTS:-}; do basename "$e"; done) > inputs-sha256.txt )
active="$( ( set -a; . ~/.config/posttrain/credentials/dstack.env; DSTACK_PROJECT=main; set +a; COLUMNS=250 "$dstack" ps ) | grep -Ev 'exited|failed|terminated|stopped|no offers|done|aborted|NAME|^ +\(|^ *$' || true)"
[[ -z "$active" ]] || { echo "dstack has active runs; refusing:"; echo "$active"; exit 1; } >&2
gpu="$("$ws" 'nvidia-smi --query-compute-apps=pid --format=csv,noheader')"
[[ -z "$gpu" ]] || { echo "GPU busy; refusing" >&2; exit 1; }
inputs=""
for extra in ${SDPO_AB_INPUTS:-}; do inputs+="  - $stage/$(basename "$extra"):/stage/$(basename "$extra")"$'\n'; done
args="$(printf ' %q' "$@")"
cat > "$stage/task.dstack.yml" <<YML
type: task
name: $folder
image: registry.lan/carbonteq/posttrain-lab/posttrain-lab/posttrain-job@sha256:e976e727786135f851f4cc96356ebff1f57c75ca5f2d36a2c6aacbf6b2822337
shell: /bin/bash
env:
  - PYTHONPATH=/env/environments/automationbench_v1/src:/code/scripts/qualification:/workspace/driver:/workspace/trl:/workspace/kernel-deps:/workspace/training-deps
  - VLLM_BATCH_INVARIANT=${SDPO_AB_BATCH_INVARIANT:-1}
  - VLLM_WORKER_MULTIPROC_METHOD=spawn
  - VLLM_GDN_DECODE_KERNEL=triton
  - POSTTRAIN_VERIFIERS_PREINSTALLED=1
  - POSTTRAIN_QUALIFICATION_TRACE_SECONDS=0
  - HF_HOME=/hf
  - HF_HUB_OFFLINE=1
  - TRANSFORMERS_OFFLINE=1
  - WANDB_DISABLED=true
  - DO_NOT_TRACK=1
  - HF_HUB_DISABLE_TELEMETRY=1
  - TOKENIZERS_PARALLELISM=false
  - TRITON_CACHE_DIR=/triton
  - VLLM_CACHE_ROOT=/vllm-cache
  - SDPO_AB_CODE_SHA=$sha
  - SDPO_AB_ENV_COMMIT=$(cat "$stage/env-commit.txt")
  - SDPO_AB_WORLD_DATE=${SDPO_AB_WORLD_DATE:-}
  - TZ=UTC
commands:
  - mkdir -p /code /env /evidence/$folder && tar xzf /stage/code.tgz -C /code && cat /stage/env.tgz.part* > /tmp/env.tgz && (cd /stage && sha256sum -c inputs-sha256.txt --ignore-missing) && echo "\$(grep ' ./env.tgz' /stage/inputs-sha256.txt | cut -d' ' -f1)  /tmp/env.tgz" | sha256sum -c && tar xzf /tmp/env.tgz -C /env
  - cp /stage/inputs-sha256.txt /stage/env-commit.txt /evidence/$folder/ && echo $sha > /evidence/$folder/code-sha.txt
  - cd /evidence/$folder && /opt/posttrain/venv/bin/python -u$args 2>&1 | tee -a /evidence/$folder/stdout.log; exit \${PIPESTATUS[0]}
files:
  - $stage/code.tgz:/stage/code.tgz
$(for p in $envparts; do echo "  - $stage/$p:/stage/$p"; done)
  - $stage/inputs-sha256.txt:/stage/inputs-sha256.txt
  - $stage/env-commit.txt:/stage/env-commit.txt
$inputs
volumes:
  - /var/lib/posttrain/cache/huggingface:/hf
  - /var/lib/posttrain/qualifications/self-campaign-2048-20261017-r5:/workspace
  - /var/lib/posttrain/qualifications/sdpo-ab-runs:/evidence
  - /var/lib/posttrain/cache/triton:/triton
  - /var/lib/posttrain/cache/vllm:/vllm-cache
resources:
  cpu: 8..
  memory: 32GB..
  shm_size: 16GB
  gpu:
    vendor: nvidia
    count: 1
    memory: 96GB..
fleets:
  - local-gpu-workers
max_duration: $duration
stop_duration: 2m
retry: false
YML
echo "environment commit: $envcommit -> $(cat "$stage/env-commit.txt")"
( cd "$stage"; set -a; . ~/.config/posttrain/credentials/dstack.env; DSTACK_PROJECT=main; set +a; "$dstack" apply -f task.dstack.yml -y -d ) 2>&1 | tail -20
