import type { RolloutTimeView, TraceTimeline, TraceTimelineSegment, TraceTiming } from '../lib/api';

/** GPU inference vs CPU harness work: one colour per phase, shared by both views. */
const PHASES = [
  { key: 'inference', label: 'Inference', color: '#6f96a2', where: 'GPU' },
  { key: 'tools', label: 'Tools', color: '#b98368', where: 'CPU' },
  { key: 'setup', label: 'Setup', color: '#c8c1b5', where: 'CPU' },
  { key: 'scoring', label: 'Scoring', color: '#9d93ab', where: 'CPU' },
] as const;

type PhaseKey = (typeof PHASES)[number]['key'];

const SEGMENT_PHASE: Record<TraceTimelineSegment['kind'], PhaseKey> = {
  inference: 'inference',
  tools: 'tools',
  harness: 'tools',
  setup: 'setup',
  scoring: 'scoring',
};

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const seconds = ms / 1000;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  const whole = Math.round(seconds);
  const h = Math.floor(whole / 3600);
  const m = Math.floor((whole % 3600) / 60);
  const s = whole % 60;
  return h ? `${h}h ${String(m).padStart(2, '0')}m ${String(s).padStart(2, '0')}s` : `${m}m ${String(s).padStart(2, '0')}s`;
}

function phaseTotals(timing: TraceTiming): Record<PhaseKey, number> {
  return { inference: timing.inference_ms, tools: timing.tools_ms, setup: timing.setup_ms, scoring: timing.scoring_ms };
}

function Legend({ totals, divisor = 1, suffix }: { totals: Record<PhaseKey, number>; divisor?: number; suffix?: string }) {
  const sum = PHASES.reduce((acc, phase) => acc + totals[phase.key], 0) || 1;
  return <div className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-secondary">
    {PHASES.filter((phase) => totals[phase.key] > 0).map((phase) => <span key={phase.key} className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-[2px]" style={{ background: phase.color }} />
      {phase.label} <span className="text-muted">({phase.where})</span>
      <strong className="font-medium tabular-nums text-ink">{formatDuration(totals[phase.key] / divisor)}</strong>
      <span className="tabular-nums text-muted">{Math.round((totals[phase.key] / sum) * 100)}%</span>
    </span>)}
    {suffix && <span className="text-muted">{suffix}</span>}
  </div>;
}

/** Run-level view: real elapsed generation time, and where an average rollout spends it. */
export function RolloutTimeSummary({ view }: { view: RolloutTimeView | null }) {
  if (!view || view.state !== 'available' || !view.steps.length) return null;
  const totals = view.steps.reduce<Record<PhaseKey, number>>((acc, step) => {
    acc.inference += step.inference_ms ?? 0;
    acc.tools += step.tools_ms ?? 0;
    acc.setup += step.setup_ms ?? 0;
    acc.scoring += step.scoring_ms ?? 0;
    return acc;
  }, { inference: 0, tools: 0, setup: 0, scoring: 0 });
  const rolloutMs = PHASES.reduce((acc, phase) => acc + totals[phase.key], 0);
  if (rolloutMs <= 0) return null;
  const rollouts = Math.max(view.steps.reduce((acc, step) => acc + step.timed_rollouts, 0), 1);
  const optimizerSteps = view.steps.filter((step) => step.step != null).length;
  const elapsed = view.elapsed_ms != null && view.elapsed_ms > 0 ? view.elapsed_ms : null;
  // Summed rollout time over elapsed time: how many rollouts were running at once.
  const inFlight = elapsed ? rolloutMs / elapsed : null;
  return <section className="obs-card px-4 py-3.5" aria-label="Where rollout time goes">
    <div className="flex items-baseline justify-between gap-3">
      <div>
        <h2 className="text-[13px] font-medium">Where rollout time goes</h2>
        <p className="mt-0.5 text-[11px] text-muted">
          {rollouts.toLocaleString()} rollouts · {optimizerSteps} optimizer steps
          {inFlight != null && <> · ~{inFlight.toFixed(inFlight < 10 ? 1 : 0)} rollouts in flight on average</>}
          {view.live ? ' · live' : ''}
        </p>
      </div>
      {elapsed != null && <span className="text-right" title="First rollout start to last rollout end, per step; time between steps (training, weight sync) is excluded">
        <span className="font-mono text-lg tabular-nums text-ink">{formatDuration(elapsed)}</span>
        <span className="ml-1.5 text-[11px] text-muted">generating</span>
      </span>}
    </div>
    <div className="mt-3 flex h-7 overflow-hidden rounded-[4px] bg-subtle" role="img" aria-label="Average rollout time by phase">
      {PHASES.map((phase) => totals[phase.key] > 0 && <span
        key={phase.key}
        className="h-full"
        style={{ width: `${(totals[phase.key] / rolloutMs) * 100}%`, background: phase.color }}
        title={`${phase.label}: ${formatDuration(totals[phase.key] / rollouts)} per rollout (${Math.round((totals[phase.key] / rolloutMs) * 100)}%)`}
      />)}
    </div>
    <p className="mt-2.5 text-[11px] text-muted">Per rollout (average {formatDuration(rolloutMs / rollouts)}):</p>
    <Legend totals={totals} divisor={rollouts} />
    <p className="mt-2 text-[10px] leading-4 text-muted">Inference time includes waiting in the inference server while other rollouts are served, so it overstates GPU compute per rollout.</p>
  </section>;
}

function segmentTitle(segment: TraceTimelineSegment): string {
  const parts = [`${segment.kind === 'harness' ? 'Harness' : PHASES.find((phase) => phase.key === SEGMENT_PHASE[segment.kind])?.label}: ${formatDuration(segment.duration_ms)}`];
  if (segment.kind === 'inference') {
    if (segment.call_index != null) parts[0] = `Model call ${segment.call_index + 1}: ${formatDuration(segment.duration_ms)}`;
    if (segment.completion_tokens != null) {
      parts.push(`${segment.completion_tokens.toLocaleString()} output tokens (${(segment.completion_tokens / Math.max(segment.duration_ms / 1000, 1e-3)).toFixed(0)} tok/s)`);
    }
    if (segment.thinking_tokens != null) parts.push(`${segment.thinking_tokens.toLocaleString()} thinking tokens`);
    if (segment.prompt_tokens != null) parts.push(`${segment.prompt_tokens.toLocaleString()} prompt tokens`);
    if (segment.finish_reason) parts.push(`finish: ${segment.finish_reason}`);
  }
  if (segment.tools.length) parts.push(segment.tools.join(', '));
  return parts.join('\n');
}

function scrollToTurn(segment: TraceTimelineSegment) {
  const byNode = segment.node != null ? document.querySelector(`[data-turn-node="${segment.node}"]`) : null;
  const target = byNode ?? (segment.call_index != null ? document.getElementById(`rollout-turn-${segment.call_index}`) : null);
  target?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/** Per-trajectory view: every phase and model call in order, scaled to the rollout's duration. */
export function RolloutTimeline({ timeline }: { timeline: TraceTimeline }) {
  const span = Math.max(timeline.total_ms, ...timeline.segments.map((segment) => segment.start_ms + segment.duration_ms), 1);
  return <section className="border-b border-divider p-4" aria-labelledby="rollout-timeline-heading">
    <div className="flex items-baseline justify-between gap-3">
      <h3 id="rollout-timeline-heading" className="type-label">Where time went</h3>
      <span className="font-mono text-sm tabular-nums text-ink">{formatDuration(timeline.total_ms)}</span>
    </div>
    <div className="relative mt-2.5 h-6 overflow-hidden rounded-[4px] bg-subtle" role="img" aria-label={`Rollout timeline: ${timeline.model_calls} model calls`}>
      {timeline.segments.map((segment, index) => {
        const phase = PHASES.find((item) => item.key === SEGMENT_PHASE[segment.kind]);
        return <span
          key={`${segment.kind}-${index}`}
          className={`absolute top-0 h-full ${segment.kind === 'inference' ? 'cursor-pointer hover:brightness-90' : ''}`}
          style={{
            left: `${(segment.start_ms / span) * 100}%`,
            width: `max(1px, ${(segment.duration_ms / span) * 100}%)`,
            background: phase?.color,
            opacity: segment.kind === 'harness' ? 0.55 : 1,
            boxShadow: segment.kind === 'inference' ? 'inset -1px 0 0 rgba(255,255,255,.55)' : undefined,
          }}
          title={segmentTitle(segment)}
          onClick={segment.kind === 'inference' ? () => scrollToTurn(segment) : undefined}
        />;
      })}
    </div>
    <Legend totals={phaseTotals(timeline)} suffix={`${timeline.model_calls} model calls`} />
  </section>;
}
