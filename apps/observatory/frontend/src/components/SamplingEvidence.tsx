import type { GRPOSamplingEvidence, GRPOSamplingStep, SummaryMetric } from '../lib/api';

const CLASS_COLORS = [
  '#6d5bd0',
  '#23878a',
  '#b96d32',
  '#718b3a',
  '#3276a8',
  '#a45179',
  '#8a6c2f',
  '#596579',
];

function numeric(metric: SummaryMetric): number | null {
  return typeof metric.value === 'number' && Number.isFinite(metric.value) ? metric.value : null;
}

function count(metric: SummaryMetric): string {
  const value = numeric(metric);
  return value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: 1 });
}

function ratio(metric: SummaryMetric): string {
  const value = numeric(metric);
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`;
}

function strategyTitle(sampling: GRPOSamplingEvidence): string {
  if (sampling.strategy === 'olmo3_active') return 'OLMo active sampling';
  if (sampling.strategy === 'dynamic') return 'Dynamic sampling';
  return 'Optimizer sampling';
}

export function SamplingSummary({ sampling }: { sampling: GRPOSamplingEvidence }) {
  const olmo = sampling.strategy === 'olmo3_active';
  return <section className="obs-card mt-3 overflow-hidden" aria-label={`${strategyTitle(sampling)} evidence`}>
    <header className="flex flex-wrap items-end justify-between gap-3 border-b border-divider px-4 py-3">
      <div>
        <p className="type-eyebrow">SAMPLING POLICY</p>
        <h2 className="mt-1 font-serif text-lg font-normal">{strategyTitle(sampling)}</h2>
      </div>
      <p className="max-w-xl text-[11px] leading-4 text-muted">
        {olmo
          ? 'Candidate groups are generated until enough mixed-reward groups are retained. Candidate rejection measures selection cost; it does not describe the optimizer batch.'
          : sampling.strategy === 'dynamic'
            ? 'The sampler filters generated groups before optimization. Read retention beside rollout cost and update evidence.'
            : 'All generated groups enter the optimizer population; zero variance therefore describes the training batch.'}
      </p>
    </header>
    <div className="grid grid-cols-2 sm:grid-cols-4">
      <SamplingMetric label={olmo ? 'Candidate retention' : 'Retained fraction'} value={ratio(sampling.retained_fraction)} note={olmo ? 'generated groups kept' : 'sampler yield'} />
      <SamplingMetric label={sampling.zero_variance.label} value={ratio(sampling.zero_variance)} note={`${sampling.zero_variance_scope} population`} />
      <SamplingMetric label="Task groups" value={`${count(sampling.retained_groups)} / ${count(sampling.generated_groups)}`} note="retained / generated" />
      <SamplingMetric label="Refill rounds" value={count(sampling.generation_rounds)} note="latest optimizer step" />
    </div>
  </section>;
}

function SamplingMetric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="border-b border-r border-divider px-4 py-3 last:border-r-0">
    <span className="block text-[10px] text-muted">{label}</span>
    <strong className="mt-1 block font-serif text-xl font-normal tabular-nums">{value}</strong>
    <span className="mt-1 block text-[9px] text-muted">{note}</span>
  </div>;
}

export function SamplingDistribution({ sampling }: { sampling: GRPOSamplingEvidence }) {
  if (!sampling.steps.length) return null;
  const classes = [...new Set(sampling.steps.flatMap((step) => Object.keys(step.class_counts)))].sort();
  const colors = new Map(classes.map((classId, index) => [classId, CLASS_COLORS[index % CLASS_COLORS.length]]));
  return <section className="obs-card mt-3 overflow-hidden" aria-label="Adaptive sampling distribution by step">
    <header className="flex flex-wrap items-end justify-between gap-3 border-b border-divider px-4 py-3">
      <div>
        <p className="type-eyebrow">CONTROLLER EVIDENCE</p>
        <h2 className="mt-1 font-serif text-lg font-normal">Candidate allocation by optimizer step</h2>
      </div>
      <p className="max-w-xl text-[11px] leading-4 text-muted">Each bar is the complete candidate population selected across initial generation and refill rounds. Colors show class allocation; counts distinguish discovery, revisits, and degraded duplicate fallback.</p>
    </header>
    <div className="flex flex-wrap gap-x-4 gap-y-1 border-b border-divider bg-subtle/45 px-4 py-2">
      {classes.map((classId) => <span key={classId} className="inline-flex items-center gap-1.5 text-[10px] text-secondary"><span className="h-2 w-2 rounded-sm" style={{ backgroundColor: colors.get(classId) }} />{classId}</span>)}
    </div>
    <div className="space-y-3 px-4 py-4">
      {sampling.steps.map((step) => <StepDistribution key={step.step} step={step} classes={classes} colors={colors} />)}
    </div>
  </section>;
}

function StepDistribution({ step, classes, colors }: {
  step: GRPOSamplingStep;
  classes: string[];
  colors: Map<string, string>;
}) {
  const denominator = Math.max(1, step.candidate_groups);
  const revisited = Math.max(0, step.candidate_groups - step.new_tasks);
  const description = classes
    .filter((classId) => (step.class_counts[classId] ?? 0) > 0)
    .map((classId) => `${classId} ${step.class_counts[classId]}`)
    .join(', ');
  return <div className="grid gap-2 sm:grid-cols-[72px_minmax(0,1fr)_310px] sm:items-center">
    <div>
      <strong className="block text-[11px] font-medium">Step {step.step}</strong>
      <span className="text-[9px] text-muted">{step.refill_rounds || 1} {step.refill_rounds === 1 ? 'round' : 'rounds'}</span>
    </div>
    <div className="flex h-5 overflow-hidden rounded-[3px] bg-subtle" role="img" aria-label={`Step ${step.step}: ${description}`}>
      {classes.map((classId) => {
        const value = step.class_counts[classId] ?? 0;
        return value > 0 ? <span key={classId} title={`${classId}: ${value} candidate groups`} style={{ width: `${value / denominator * 100}%`, backgroundColor: colors.get(classId) }} /> : null;
      })}
    </div>
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-[9px] text-muted">
      <span><strong className="text-ink">{step.candidate_groups}</strong> candidates</span>
      <span><strong className="text-ink">{step.new_tasks}</strong> new</span>
      <span><strong className="text-ink">{revisited}</strong> revisited</span>
      <span><strong className="text-ink">{step.discovery_fulfilled}/{step.discovery_reserved}</strong> discovery</span>
      <span className={step.duplicate_fallbacks ? 'text-amber-700' : ''}><strong>{step.duplicate_fallbacks}</strong> fallback repeats</span>
    </div>
  </div>;
}
