import { useState } from 'react';

import type {
  PhaseMetricAggregate,
  RuntimePhaseInterval,
  RuntimePhaseSegment,
  RuntimePhaseSummary,
  SystemMetrics,
} from '../lib/api';

const phaseColors: Record<string, string> = {
  model_loading: '#6557d2',
  model_offloading: '#7c6371',
  data_preparation: '#2f7d73',
  runtime_initialization: '#3b82a0',
  benchmark_warmup: '#c47a28',
  benchmark_measurement: '#28745c',
  runtime_cleanup: '#8b6977',
  rollout: '#d66a45',
  reward_scoring: '#b45f8c',
  teacher_scoring: '#9b6a34',
  actor_update: '#4e66c8',
  evaluation: '#38866f',
  artifact_export: '#92724b',
  backend_execution: '#58677a',
  operation: '#72717a',
  unclassified: '#9d9aa3',
};

function phaseColor(phase: string): string {
  return phaseColors[phase] ?? '#6d5f91';
}

function metric(
  values: PhaseMetricAggregate[],
  name: string,
): PhaseMetricAggregate | undefined {
  return values.find((value) => value.metric === name);
}

function formatBytes(value: number | null | undefined): string {
  if (value == null) return '—';
  const gib = value / 1024 ** 3;
  return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function phasePressure(peak: number, capacity: number): {
  label: string;
  className: string;
} {
  const ratio = peak / capacity;
  if (ratio >= 1) return { label: 'Exceeds VRAM capacity', className: 'text-rose-700' };
  if (ratio >= 0.95) return { label: 'VRAM capacity saturated', className: 'text-rose-700' };
  if (ratio >= 0.8) return { label: 'VRAM capacity constrained', className: 'text-amber-700' };
  return { label: 'VRAM capacity comfortable', className: 'text-emerald-700' };
}

function targetSummary(system: SystemMetrics): string {
  if (!system.execution_targets.length) return 'Execution target was not retained';
  const complete = system.execution_targets.filter((target) => target.state === 'complete');
  if (!complete.length) {
    return `${system.execution_targets.map((target) => target.selection_id).join(', ')} · capacity missing from historical snapshot`;
  }
  const first = complete[0];
  const deviceCount = first.device_count ?? 1;
  const deviceLabel = `${deviceCount} × ${first.device_class ?? 'device'}`;
  const memoryLabel = first.memory_bytes_per_device
    ? `${formatBytes(first.memory_bytes_per_device)} per device`
    : 'capacity not recorded';
  const extra = complete.length > 1 ? ` · ${complete.length} role-specific targets` : '';
  return `${deviceLabel} · ${memoryLabel}${extra}`;
}

function segmentTitle(segment: RuntimePhaseSegment): string {
  const vram = metric(segment.metrics, 'system/gpu_vram_used_bytes');
  const utilization = metric(segment.metrics, 'system/gpu_utilization');
  return [
    segment.label,
    formatDuration(segment.duration_s),
    utilization ? `GPU active ${utilization.mean.toFixed(1)}%` : 'GPU utilization not sampled',
    vram ? `average ${formatBytes(vram.mean)}` : 'VRAM not sampled',
    vram ? `peak ${formatBytes(vram.peak)}` : null,
    `${segment.sample_count} host sample${segment.sample_count === 1 ? '' : 's'}`,
  ].filter(Boolean).join(' · ');
}

function PhaseSummaryCard({
  phase,
  capacity,
  partial,
}: {
  phase: RuntimePhaseSummary;
  capacity: number | null;
  partial: boolean;
}) {
  const vram = metric(phase.metrics, 'system/gpu_vram_used_bytes');
  const utilization = metric(phase.metrics, 'system/gpu_utilization');
  const pressure = vram && capacity ? phasePressure(vram.peak, capacity) : null;
  return <article className="min-w-0 border-l-2 pl-3" style={{ borderColor: phaseColor(phase.phase) }}>
    <div className="flex items-baseline justify-between gap-3">
      <h3 className="truncate text-[11px] font-medium text-ink">{phase.label}</h3>
      <span className="shrink-0 text-[10px] text-muted">{partial ? 'recorded ' : ''}{formatDuration(phase.duration_s)}</span>
    </div>
    <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
      <div><dt className="text-muted">Average VRAM</dt><dd className="mt-0.5 font-medium text-secondary">{formatBytes(vram?.mean)}</dd></div>
      <div><dt className="text-muted">Peak VRAM</dt><dd className="mt-0.5 font-medium text-secondary">{formatBytes(vram?.peak)}</dd></div>
      <div><dt className="text-muted">GPU active</dt><dd className="mt-0.5 font-medium text-secondary">{utilization ? `${utilization.mean.toFixed(1)}%` : '—'}</dd></div>
      <div><dt className="text-muted">VRAM headroom</dt><dd className={`mt-0.5 font-medium ${pressure?.className ?? 'text-secondary'}`}>{vram && capacity ? `${Math.max(0, (1 - vram.peak / capacity) * 100).toFixed(1)}% · ${pressure?.label}` : '—'}</dd></div>
    </dl>
    <p className={`mt-2 text-[9px] ${partial ? 'text-amber-700' : 'text-muted'}`}>{phase.occurrences} recorded window{phase.occurrences === 1 ? '' : 's'} · {phase.sample_count} host samples{partial ? ' · partial event coverage' : ''}</p>
  </article>;
}

function CapacityView({ system }: { system: SystemMetrics }) {
  const capacity = system.vram_capacity_bytes;
  if (system.vram_capacity_state !== 'available' || capacity == null) {
    const reason = system.vram_capacity_state === 'ambiguous'
      ? 'The run names execution targets with different capacities, so host-level memory cannot be assigned to one trustworthy denominator.'
      : 'The run does not contain a complete immutable execution-target capacity. Historical target identifiers remain visible, but Observatory will not infer bytes from their names.';
    return <div className="border-b border-divider px-4 py-5">
      <div className="rounded-[5px] border border-dashed border-amber-300 bg-[#fffaf1] p-4">
        <h3 className="text-[12px] font-medium text-ink">Capacity comparison unavailable</h3>
        <p className="mt-1 max-w-3xl text-[11px] leading-5 text-secondary">{reason}</p>
      </div>
    </div>;
  }

  const chartX = 210;
  const chartWidth = 610;
  const rowHeight = 42;
  const chartTop = 38;
  const height = chartTop + system.phase_summary.length * rowHeight + 34;
  return <div className="border-b border-divider px-4 py-3">
    <svg
      role="img"
      aria-label="Phase GPU memory against declared hardware capacity"
      className="block h-auto w-full"
      viewBox={`0 0 1000 ${height}`}
    >
      <desc>Each row is one runtime phase. The filled bar is mean allocated GPU memory, the vertical marker is peak memory, and the full track is declared aggregate execution-target capacity.</desc>
      <text x={chartX} y="15" fill="#827b85" fontSize="10">0 GiB</text>
      <text x={chartX + chartWidth} y="15" fill="#827b85" fontSize="10" textAnchor="end">{formatBytes(capacity)} declared capacity</text>
      {system.phase_summary.map((phase, index) => {
        const vram = metric(phase.metrics, 'system/gpu_vram_used_bytes');
        const meanRatio = vram ? Math.min(vram.mean / capacity, 1) : 0;
        const peakRatio = vram ? Math.min(vram.peak / capacity, 1) : 0;
        const rawPeakRatio = vram ? vram.peak / capacity : 0;
        const y = chartTop + index * rowHeight;
        const color = phaseColor(phase.phase);
        const pressure = vram ? phasePressure(vram.peak, capacity) : null;
        return <g key={phase.phase}>
          <title>{vram ? `${phase.label}: mean ${formatBytes(vram.mean)}, peak ${formatBytes(vram.peak)}, ${(rawPeakRatio * 100).toFixed(1)}% of declared capacity` : `${phase.label}: no GPU memory samples`}</title>
          <text x="0" y={y + 11} fill="#332f35" fontSize="11" fontWeight="500">{phase.label}</text>
          <text x="0" y={y + 27} fill="#827b85" fontSize="9">{formatDuration(phase.duration_s)} · {phase.sample_count} samples</text>
          <rect x={chartX} y={y} width={chartWidth} height="24" rx="3" fill="#f0ede9" stroke="#ded9d3" />
          {vram && <rect x={chartX} y={y} width={chartWidth * meanRatio} height="24" rx="3" fill={color} opacity="0.72" />}
          {vram && <line x1={chartX + chartWidth * peakRatio} x2={chartX + chartWidth * peakRatio} y1={y - 3} y2={y + 27} stroke={rawPeakRatio >= 0.95 ? '#be3348' : color} strokeWidth="2" />}
          <text x={chartX + chartWidth + 18} y={y + 10} fill="#332f35" fontSize="10">{vram ? `${(rawPeakRatio * 100).toFixed(1)}% peak` : 'not sampled'}</text>
          <text x={chartX + chartWidth + 18} y={y + 25} fill={pressure?.className.includes('rose') ? '#b4233a' : pressure?.className.includes('amber') ? '#a05b12' : '#28745c'} fontSize="9">{pressure?.label ?? 'Missing evidence'}</text>
        </g>;
      })}
      <line x1={chartX + chartWidth} x2={chartX + chartWidth} y1="25" y2={height - 22} stroke="#8d8490" strokeDasharray="3 3" />
    </svg>
  </div>;
}

function phaseLanes(intervals: RuntimePhaseInterval[]): Array<{
  phase: string;
  label: string;
  group: string;
  groupLabel: string;
  intervals: RuntimePhaseInterval[];
}> {
  const lanes = new Map<string, {
    phase: string;
    label: string;
    group: string;
    groupLabel: string;
    intervals: RuntimePhaseInterval[];
  }>();
  intervals.forEach((interval) => {
    const lane = lanes.get(interval.phase) ?? {
      phase: interval.phase,
      label: interval.label,
      group: interval.group ?? 'other',
      groupLabel: interval.group_label ?? 'Other',
      intervals: [],
    };
    lane.intervals.push(interval);
    lanes.set(interval.phase, lane);
  });
  return [...lanes.values()];
}

function phaseGroups(phases: RuntimePhaseSummary[]): Array<{
  group: string;
  label: string;
  phases: RuntimePhaseSummary[];
}> {
  const groups = new Map<string, { group: string; label: string; phases: RuntimePhaseSummary[] }>();
  phases.forEach((phase) => {
    const groupKey = phase.group ?? 'other';
    const item = groups.get(groupKey) ?? {
      group: groupKey,
      label: phase.group_label ?? 'Other',
      phases: [],
    };
    item.phases.push(phase);
    groups.set(groupKey, item);
  });
  return [...groups.values()];
}

function InferenceDetails({ system }: { system: SystemMetrics }) {
  const timing = system.inference_timing;
  const runtime = system.backend_runtime;
  const rollout = system.phase_summary.find((phase) => phase.phase === 'rollout');
  if (!timing && !runtime && !rollout) return null;
  const peak = runtime?.kv_cache_peak_usage_ratio ?? null;
  const rolloutUtilization = rollout ? metric(rollout.metrics, 'system/gpu_utilization') : null;
  const rolloutVram = rollout ? metric(rollout.metrics, 'system/gpu_vram_used_bytes') : null;
  const rolloutVramRatio = rolloutVram && system.vram_capacity_bytes
    ? rolloutVram.peak / system.vram_capacity_bytes
    : null;
  const hasRolloutThroughput = runtime?.rollout_tokens_per_second_latest != null
    || runtime?.rollout_tokens_per_second_mean != null
    || runtime?.rollout_seconds_latest != null;
  const hasRuntimeConfiguration = runtime != null && [
    runtime.environment_concurrency,
    runtime.inference_sequence_cap,
    runtime.rollouts_per_prompt,
    runtime.rollouts_per_update,
  ].some((value) => value != null);
  return <section aria-label="Inference details" className="border-b border-divider bg-[#fbfaf8] px-4 py-4">
    <div className="flex flex-wrap items-baseline justify-between gap-2">
      <div>
        <h3 className="text-[12px] font-medium text-ink">Inference details</h3>
        <p className="mt-1 text-[10px] text-muted">{rollout ? 'GPU activity measures compute busy time; VRAM and KV-cache pressure are separate capacity signals.' : 'Per-request timings overlap at concurrency and do not sum to measured run wall time.'}</p>
      </div>
      {timing && <span className="text-[10px] text-muted">{timing.requests} measured requests</span>}
    </div>
    <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {timing && timing.stages.map((stage) => <article key={stage.stage} className="rounded-[4px] border border-divider bg-surface p-3">
          <div className="flex items-baseline justify-between gap-2">
            <h4 className="text-[11px] font-medium">{stage.label}</h4>
            <span className="text-[9px] text-muted">{stage.samples} samples</span>
          </div>
          <strong className="mt-2 block font-serif text-xl font-normal">{stage.p50_ms.toFixed(stage.p50_ms < 10 ? 1 : 0)} ms</strong>
          <p className="mt-1 text-[9px] text-muted">p50 · p95 {stage.p95_ms.toFixed(stage.p95_ms < 10 ? 1 : 0)} ms · mean {stage.mean_ms.toFixed(stage.mean_ms < 10 ? 1 : 0)} ms</p>
        </article>)}
      {rollout && <article className="rounded-[4px] border border-divider bg-surface p-3">
        <div className="flex items-baseline justify-between gap-2"><h4 className="text-[11px] font-medium">Rollout compute</h4><span className={`text-[9px] ${system.phase_state === 'partial' ? 'text-amber-700' : 'text-muted'}`}>{rollout.occurrences} windows</span></div>
        <strong className="mt-2 block font-serif text-xl font-normal">{rolloutUtilization ? `${rolloutUtilization.mean.toFixed(1)}% average` : 'Not sampled'}</strong>
        <p className="mt-1 text-[9px] text-secondary">{rolloutUtilization ? `${rolloutUtilization.peak.toFixed(1)}% peak GPU activity` : 'GPU activity unavailable'}</p>
        <p className="mt-1 text-[9px] text-muted">{rolloutVramRatio == null ? 'Peak VRAM unavailable' : `${(rolloutVramRatio * 100).toFixed(1)}% peak VRAM · ${phasePressure(rolloutVram?.peak ?? 0, system.vram_capacity_bytes ?? 1).label}`}</p>
        {system.phase_state === 'partial' && <p className="mt-2 text-[9px] text-amber-700">Partial phase events; do not treat this as the full-run average.</p>}
      </article>}
      {runtime && hasRolloutThroughput && <article className="rounded-[4px] border border-divider bg-surface p-3">
        <div className="flex items-baseline justify-between gap-2"><h4 className="text-[11px] font-medium">Rollout throughput</h4><span className="text-[9px] text-muted">{runtime.rollout_samples} steps</span></div>
        <strong className="mt-2 block font-serif text-xl font-normal">{runtime.rollout_tokens_per_second_latest == null ? '—' : `${runtime.rollout_tokens_per_second_latest.toFixed(0)} tok/s`}</strong>
        <p className="mt-1 text-[9px] text-secondary">latest · mean {runtime.rollout_tokens_per_second_mean == null ? '—' : `${runtime.rollout_tokens_per_second_mean.toFixed(0)} tok/s`}</p>
        <p className="mt-1 text-[9px] text-muted">{runtime.rollout_seconds_latest == null ? 'Latest batch duration unavailable' : `${formatDuration(runtime.rollout_seconds_latest)} latest batch · ${runtime.rollout_seconds_mean == null ? 'mean unavailable' : `${formatDuration(runtime.rollout_seconds_mean)} mean`}`}</p>
      </article>}
      {runtime?.mtp_selected && <article className="rounded-[4px] border border-divider bg-surface p-3">
        <div className="flex items-baseline justify-between gap-2"><h4 className="text-[11px] font-medium">MTP acceleration</h4><span className="text-[9px] text-muted">{runtime.mtp_samples} steps</span></div>
        <strong className="mt-2 block font-serif text-xl font-normal">{runtime.mtp_acceptance_rate == null ? 'Enabled' : `${(runtime.mtp_acceptance_rate * 100).toFixed(1)}% acceptance`}</strong>
        <p className="mt-1 text-[9px] text-secondary">{runtime.mtp_accepted_length == null ? 'Accepted length unavailable' : `${runtime.mtp_accepted_length.toFixed(2)} accepted tokens per verification cycle`}</p>
        <p className="mt-1 text-[9px] text-muted">Rollout optimization only; no MTP training objective.</p>
      </article>}
      {runtime && <article className="rounded-[4px] border border-divider bg-surface p-3">
        <div className="flex items-baseline justify-between gap-2">
          <h4 className="text-[11px] font-medium">KV-cache pressure</h4>
          <span className="text-[9px] text-muted">{runtime.kv_cache_samples} step samples</span>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-[#e8e3dd]">
          <div className="h-full rounded-full bg-violet-600" style={{ width: `${Math.min((peak ?? 0) * 100, 100)}%` }} />
        </div>
        <p className="mt-2 text-[10px] text-secondary">{peak == null ? 'Peak not recorded' : `${(peak * 100).toFixed(1)}% peak KV-cache occupancy`}</p>
        <p className="mt-1 text-[9px] text-muted">{runtime.kv_cache_capacity_tokens == null ? 'Capacity not reported by backend' : `${Math.round(runtime.kv_cache_capacity_tokens).toLocaleString()} token group-aware capacity`}</p>
      </article>}
    </div>
    {runtime && hasRuntimeConfiguration && <dl className="mt-3 grid gap-x-5 gap-y-2 border-t border-divider pt-3 text-[9px] sm:grid-cols-2 xl:grid-cols-4">
      <div><dt className="uppercase tracking-[.06em] text-muted">Environment concurrency</dt><dd className="mt-0.5 text-secondary">{runtime.environment_concurrency ?? '—'}</dd></div>
      <div><dt className="uppercase tracking-[.06em] text-muted">vLLM sequence cap</dt><dd className="mt-0.5 text-secondary">{runtime.inference_sequence_cap ?? '—'}</dd></div>
      <div><dt className="uppercase tracking-[.06em] text-muted">Rollouts / prompt</dt><dd className="mt-0.5 text-secondary">{runtime.rollouts_per_prompt ?? '—'}</dd></div>
      <div><dt className="uppercase tracking-[.06em] text-muted">Rollouts / update</dt><dd className="mt-0.5 text-secondary">{runtime.rollouts_per_update ?? '—'}</dd></div>
    </dl>}
  </section>;
}

function TimelineView({ system }: { system: SystemMetrics }) {
  const totalSeconds = Math.max(
    ...system.phase_intervals.map((interval) => interval.end_offset_s),
    ...system.phase_segments.map((segment) => segment.end_offset_s),
    1,
  );
  const memory = system.groups
    .flatMap((group) => group.series)
    .find((series) => series.name === 'system/gpu_vram_used_bytes');
  const utilization = system.groups
    .flatMap((group) => group.series)
    .find((series) => series.name === 'system/gpu_utilization');
  const kvCache = system.groups
    .flatMap((group) => group.series)
    .find((series) => series.name === 'serve/backend/kv_cache_usage_ratio');
  const observedPeak = system.vram_observed_peak_bytes;
  const memoryReference = Math.max(system.vram_capacity_bytes ?? 0, observedPeak ?? 0, 1);
  const runStartedAt = Date.parse(system.window_started_at);
  const memoryPoints = memory?.points.flatMap((point) => {
    if (!point.observed_at) return [];
    const offset = (Date.parse(point.observed_at) - runStartedAt) / 1000;
    if (!Number.isFinite(offset) || offset < 0 || offset > totalSeconds) return [];
    return [{ offset, value: point.value }];
  }) ?? [];
  const utilizationPoints = utilization?.points.flatMap((point) => {
    if (!point.observed_at) return [];
    const offset = (Date.parse(point.observed_at) - runStartedAt) / 1000;
    if (!Number.isFinite(offset) || offset < 0 || offset > totalSeconds) return [];
    return [{ offset, value: Math.max(0, Math.min(point.value, 100)) }];
  }) ?? [];
  const kvCachePoints = kvCache?.points.flatMap((point) => {
    if (!point.observed_at) return [];
    const offset = (Date.parse(point.observed_at) - runStartedAt) / 1000;
    if (!Number.isFinite(offset) || offset < 0 || offset > totalSeconds) return [];
    return [{ offset, value: Math.max(0, Math.min(point.value, 1)) }];
  }) ?? [];
  const lanes = phaseLanes(system.phase_intervals);
  const chartX = 190;
  const chartWidth = 770;
  const utilizationTop = 30;
  const utilizationHeight = 58;
  const memoryTop = 115;
  const memoryHeight = 58;
  const kvCacheTop = 200;
  const kvCacheHeight = 58;
  const laneTop = kvCachePoints.length ? 285 : 205;
  const laneHeight = 30;
  const laneGroups = [...new Map(lanes.map((lane) => [lane.group, lane.groupLabel])).entries()];
  const laneRows: Array<
    | { kind: 'group'; key: string; label: string; y: number }
    | { kind: 'lane'; key: string; lane: (typeof lanes)[number]; y: number }
  > = [];
  let nextLaneY = laneTop;
  laneGroups.forEach(([group, groupLabel]) => {
    laneRows.push({ kind: 'group', key: group, label: groupLabel, y: nextLaneY });
    nextLaneY += 22;
    lanes.filter((lane) => lane.group === group).forEach((lane) => {
      laneRows.push({ kind: 'lane', key: lane.phase, lane, y: nextLaneY });
      nextLaneY += laneHeight;
    });
  });
  const height = nextLaneY + 40;
  const memoryPolyline = memoryPoints.map((point) => {
    const x = chartX + (point.offset / totalSeconds) * chartWidth;
    const y = memoryTop + memoryHeight - Math.min(point.value / memoryReference, 1) * memoryHeight;
    return `${x},${y}`;
  }).join(' ');
  const utilizationPolyline = utilizationPoints.map((point) => {
    const x = chartX + (point.offset / totalSeconds) * chartWidth;
    const y = utilizationTop + utilizationHeight - (point.value / 100) * utilizationHeight;
    return `${x},${y}`;
  }).join(' ');
  const kvCachePolyline = kvCachePoints.map((point) => {
    const x = chartX + (point.offset / totalSeconds) * chartWidth;
    const y = kvCacheTop + kvCacheHeight - point.value * kvCacheHeight;
    return `${x},${y}`;
  }).join(' ');

  return <div className="border-b border-divider px-4 py-3">
    <svg
      role="img"
      aria-label="Runtime phase and GPU utilization timeline"
      className="block h-auto w-full"
      viewBox={`0 0 1000 ${height}`}
    >
      <desc>GPU utilization, allocated memory, and backend KV-cache pressure are plotted over elapsed time when available. Semantic phase lanes are grouped so one-time startup and finalization costs remain distinct from inference or training execution.</desc>
      <rect x={chartX} y={utilizationTop} width={chartWidth} height={utilizationHeight} fill="#edf5f1" rx="3" />
      {[0.25, 0.5, 0.75].map((ratio) => <line key={`utilization-${ratio}`} x1={chartX} x2={chartX + chartWidth} y1={utilizationTop + utilizationHeight * (1 - ratio)} y2={utilizationTop + utilizationHeight * (1 - ratio)} stroke="#d7e6df" strokeDasharray="2 4" />)}
      <text x="0" y={utilizationTop + 14} fill="#332f35" fontSize="11" fontWeight="500">GPU utilization</text>
      <text x="0" y={utilizationTop + 30} fill="#827b85" fontSize="9">{utilizationPoints.length} timestamped samples</text>
      {utilizationPolyline && <polyline points={utilizationPolyline} fill="none" stroke="#28745c" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />}
      {utilizationPoints.map((point) => {
        const x = chartX + (point.offset / totalSeconds) * chartWidth;
        const y = utilizationTop + utilizationHeight - (point.value / 100) * utilizationHeight;
        return <circle key={`utilization-${point.offset}:${point.value}`} cx={x} cy={y} r="2.5" fill="#fff" stroke="#28745c"><title>{formatDuration(point.offset)} · {point.value.toFixed(1)}% GPU utilization</title></circle>;
      })}
      <text x={chartX + chartWidth} y={utilizationTop - 8} fill="#827b85" fontSize="9" textAnchor="end">100% busy</text>
      <rect x={chartX} y={memoryTop} width={chartWidth} height={memoryHeight} fill="#f3f1ee" rx="3" />
      {[0.25, 0.5, 0.75].map((ratio) => <line key={`memory-${ratio}`} x1={chartX} x2={chartX + chartWidth} y1={memoryTop + memoryHeight * (1 - ratio)} y2={memoryTop + memoryHeight * (1 - ratio)} stroke="#ded9d3" strokeDasharray="2 4" />)}
      <text x="0" y={memoryTop + 14} fill="#332f35" fontSize="11" fontWeight="500">Allocated VRAM</text>
      <text x="0" y={memoryTop + 30} fill="#827b85" fontSize="9">{memoryPoints.length} timestamped samples</text>
      {memoryPolyline && <polyline points={memoryPolyline} fill="none" stroke="#6557d2" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />}
      {memoryPoints.map((point) => {
        const x = chartX + (point.offset / totalSeconds) * chartWidth;
        const y = memoryTop + memoryHeight - Math.min(point.value / memoryReference, 1) * memoryHeight;
        return <circle key={`${point.offset}:${point.value}`} cx={x} cy={y} r="2.5" fill="#fff" stroke="#6557d2"><title>{formatDuration(point.offset)} · {formatBytes(point.value)}</title></circle>;
      })}
      <text x={chartX + chartWidth} y={memoryTop - 8} fill="#827b85" fontSize="9" textAnchor="end">{system.vram_capacity_bytes ? `capacity ${formatBytes(system.vram_capacity_bytes)}` : `observed peak ${formatBytes(observedPeak)}`}</text>
      {kvCachePoints.length > 0 && <>
        <rect x={chartX} y={kvCacheTop} width={chartWidth} height={kvCacheHeight} fill="#f3effa" rx="3" />
        {[0.25, 0.5, 0.75].map((ratio) => <line key={`kv-${ratio}`} x1={chartX} x2={chartX + chartWidth} y1={kvCacheTop + kvCacheHeight * (1 - ratio)} y2={kvCacheTop + kvCacheHeight * (1 - ratio)} stroke="#e0d8ef" strokeDasharray="2 4" />)}
        <text x="0" y={kvCacheTop + 14} fill="#332f35" fontSize="11" fontWeight="500">KV-cache pressure</text>
        <text x="0" y={kvCacheTop + 30} fill="#827b85" fontSize="9">{kvCachePoints.length} scheduler samples</text>
        <polyline points={kvCachePolyline} fill="none" stroke="#7654b3" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        {kvCachePoints.map((point) => {
          const x = chartX + (point.offset / totalSeconds) * chartWidth;
          const y = kvCacheTop + kvCacheHeight - point.value * kvCacheHeight;
          return <circle key={`kv-${point.offset}:${point.value}`} cx={x} cy={y} r="2.5" fill="#fff" stroke="#7654b3"><title>{formatDuration(point.offset)} · {(point.value * 100).toFixed(1)}% KV-cache usage</title></circle>;
        })}
        <text x={chartX + chartWidth} y={kvCacheTop - 8} fill="#827b85" fontSize="9" textAnchor="end">100% scheduler capacity</text>
      </>}
      {laneRows.map((row) => {
        if (row.kind === 'group') {
          return <g key={`group-${row.key}`}>
            <text x="0" y={row.y + 13} fill="#332f35" fontSize="10" fontWeight="600" letterSpacing="0.4">{row.label.toUpperCase()}</text>
            <line x1={chartX} x2={chartX + chartWidth} y1={row.y + 9} y2={row.y + 9} stroke="#cbc4ce" strokeDasharray="3 4" />
          </g>;
        }
        const { lane, y } = row;
        const color = phaseColor(lane.phase);
        return <g key={row.key}>
          <text x="12" y={y + 16} fill="#57515e" fontSize="10">{lane.label}</text>
          <line x1={chartX} x2={chartX + chartWidth} y1={y + 11} y2={y + 11} stroke="#e7e2dd" />
          {lane.intervals.map((interval) => {
            const x = chartX + (interval.start_offset_s / totalSeconds) * chartWidth;
            const width = Math.max(((interval.end_offset_s - interval.start_offset_s) / totalSeconds) * chartWidth, 2);
            return <rect key={interval.phase_id} x={x} y={y + 2} width={width} height="18" rx="3" fill={color} opacity={lane.phase === 'operation' ? 0.18 : 0.72} stroke={interval.status === 'failed' ? '#be3348' : color}>
              <title>{interval.label} · {formatDuration(interval.duration_s)} · {interval.status}</title>
            </rect>;
          })}
        </g>;
      })}
      <line x1={chartX} x2={chartX + chartWidth} y1={height - 28} y2={height - 28} stroke="#aaa3ac" />
      <text x={chartX} y={height - 10} fill="#827b85" fontSize="10">0s</text>
      <text x={chartX + chartWidth} y={height - 10} fill="#827b85" fontSize="10" textAnchor="end">{formatDuration(totalSeconds)}</text>
    </svg>
  </div>;
}

export function PhaseMemoryTimeline({ system }: { system: SystemMetrics }) {
  const [view, setView] = useState<'capacity' | 'timeline'>('capacity');
  if (system.phase_state === 'unavailable' || !system.phase_segments.length) {
    return <section aria-label="Runtime phase profile" className="mt-5 rounded-[5px] border border-dashed border-divider px-4 py-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-[13px] font-medium">Runtime phase profile</h2>
        <span className="text-[10px] text-muted">Phase boundaries unavailable</span>
      </div>
      <p className="mt-1 text-xs leading-5 text-muted">This run has valid host telemetry but no provider-neutral runtime phase events. Observatory will not infer phases from metric shapes.</p>
    </section>;
  }

  return <section aria-label="Runtime phase profile" className="obs-card mt-5 overflow-hidden">
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-divider px-4 py-3">
      <div>
        <h2 className="text-[13px] font-medium">Runtime phase profile</h2>
        <p className="mt-1 text-[11px] text-muted">{targetSummary(system)}</p>
      </div>
      <div className="flex items-start gap-4">
        <div className="inline-flex rounded-[5px] border border-divider bg-subtle p-0.5" role="tablist" aria-label="Runtime phase view">
          {(['capacity', 'timeline'] as const).map((option) => <button
            key={option}
            type="button"
            role="tab"
            aria-selected={view === option}
            onClick={() => setView(option)}
            className={`rounded-[3px] px-2.5 py-1 text-[10px] capitalize ${view === option ? 'bg-surface font-medium text-violet-800 shadow-sm' : 'text-muted hover:text-ink'}`}
          >{option}</button>)}
        </div>
        <div className="text-right text-[10px] text-muted">
          <p>{system.phase_summary.length} phases · {system.sample_count} host samples</p>
          {system.phase_state === 'partial' && <p className="mt-1 text-amber-700">Partial phase coverage</p>}
        </div>
      </div>
    </div>
    {view === 'capacity' ? <CapacityView system={system} /> : <TimelineView system={system} />}
    <InferenceDetails system={system} />
    <div className="space-y-5 px-4 py-4">
      {phaseGroups(system.phase_summary).map((group) => <section key={group.group} aria-label={`${group.label} phases`}>
        <h3 className="mb-3 text-[10px] font-semibold uppercase tracking-[0.08em] text-muted">{group.label}</h3>
        <div className="grid gap-x-5 gap-y-4 sm:grid-cols-2 xl:grid-cols-4">
          {group.phases.map((phase) => <PhaseSummaryCard key={phase.phase} phase={phase} capacity={system.vram_capacity_bytes} partial={system.phase_state === 'partial'} />)}
        </div>
      </section>)}
    </div>
    {(system.unclassified_sample_count > 0 || system.phase_issues.length > 0) && <p className="border-t border-divider px-4 py-2.5 text-[10px] text-amber-700">{system.unclassified_sample_count} unclassified sample{system.unclassified_sample_count === 1 ? '' : 's'}{system.phase_issues.length ? ` · ${system.phase_issues[0]}` : ''}</p>}
  </section>;
}
