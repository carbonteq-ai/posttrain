#!/usr/bin/env bash
# This queue owns one recorded, evidence-backed generation revision only.
set -euo pipefail
capsule="$(cd "$(dirname "$0")/.." && pwd)"
baseline=/var/lib/posttrain/qualifications/luna-ssr-generation14-20261018-r1
mkdir "$capsule/evidence/queue-lock"  # Permanent duplicate guard; never auto-remove.
cd "$capsule"
sha256sum -c source-sha256.txt >/dev/null
deadline=$(( $(date +%s) + 7200 ))
while [[ ! -s "$baseline/evidence/9b-exit-code.txt" ]]; do
    if (( $(date +%s) >= deadline )); then
        echo "Baseline wait deadline exceeded; no quality job submitted" >&2
        exit 1
    fi
    sleep 15
done
for size in 4b 9b; do
    PYTHONPATH="$capsule/driver" python3 "$capsule/driver/analyze_luna_ssr_generation.py" \
        --cases "$capsule/cases" --outputs "$baseline/evidence/$size" \
        --output "$capsule/evidence/baseline-$size-audit.json"
done
bash "$capsule/driver/run_luna_ssr_quality_pilot.sh"
PYTHONPATH="$capsule/driver" python3 "$capsule/driver/analyze_luna_ssr_generation.py" \
    --cases "$capsule/cases" --outputs "$capsule/evidence/4b" \
    --output "$capsule/evidence/quality-4b-audit.json"
echo "Quality comparison finished; semantic review/native execution still pending"
