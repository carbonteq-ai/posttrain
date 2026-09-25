"""Exploration-bonus sensitivity for the yield-first, full-pool VORTEX replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import replace
from pathlib import Path

from vortex_simple import BASE_SETTINGS, run_simple

BONUSES = (0.2, 1.0, 4.0)


def experiment(snapshot, seeds=10):
    rows = []
    for source in snapshot['runs']:
        groups = [group for group in source['groups'] if group['size'] == 4 and not group['error_rollouts']]
        if len({group['task_id'] for group in groups}) < 32:
            continue
        for scenario in ('stationary', 'saturation_recovery'):
            for bonus in BONUSES:
                settings = replace(BASE_SETTINGS, uncertainty_weight=bonus)
                for seed in range(seeds):
                    row = run_simple(groups, 'soft_yield_explore', seed, scenario, settings)
                    row.update(source=source['name'], bonus=bonus)
                    rows.append(row)
            print(source['name'], scenario, flush=True)
    summaries = []
    keys = sorted({(row['source'], row['scenario'], row['bonus']) for row in rows})
    for key in keys:
        subset = [row for row in rows if (row['source'], row['scenario'], row['bonus']) == key]
        metrics = {}
        for metric in ('completed_steps', 'candidates_per_retained', 'covered_tasks',
                       'variance_per_million_output_tokens', 'late_new_fraction'):
            values = [row[metric] for row in subset]
            metrics[metric] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values)}
        summaries.append({'source': key[0], 'scenario': key[1], 'bonus': key[2], 'metrics': metrics})
    return {'bonuses': BONUSES, 'seeds': seeds, 'summaries': summaries, 'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()
    snapshot = json.loads(args.input.read_text())
    result = experiment(snapshot, args.seeds)
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['prototype_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['simulator_sha256'] = hashlib.sha256(Path(__file__).with_name('vortex_simple.py').read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
