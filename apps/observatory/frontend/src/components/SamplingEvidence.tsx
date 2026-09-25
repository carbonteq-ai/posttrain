import { useEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts/core';
import { BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import type { GRPOSamplingEvidence, GRPOSamplingStep, SummaryMetric } from '../lib/api';

echarts.use([BarChart, GridComponent, TooltipComponent, CanvasRenderer]);

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
          ? 'Each update needs a fixed number of useful groups: a task\'s rollouts must earn different rewards so the update can tell better attempts from worse ones. Groups where every rollout ties are discarded and more are sampled, so discards measure extra generation cost, not the update itself.'
          : sampling.strategy === 'dynamic'
            ? 'The sampler filters generated groups before optimization. Read retention beside rollout cost and update evidence.'
            : 'All generated groups enter the optimizer population; zero variance therefore describes the training batch.'}
      </p>
    </header>
    <div className="grid grid-cols-2 sm:grid-cols-4">
      <SamplingMetric label={olmo ? 'Useful groups' : 'Retained fraction'} value={ratio(sampling.retained_fraction)} note={olmo ? 'whole run · had reward spread, so trained on' : 'sampler yield'} />
      <SamplingMetric label={olmo ? 'Tied groups' : sampling.zero_variance.label} value={ratio(sampling.zero_variance)} note={olmo ? 'whole run · all rollouts scored the same, discarded' : `${sampling.zero_variance_scope} population`} />
      <SamplingMetric label={olmo ? 'Groups kept / sampled' : 'Task groups'} value={`${count(sampling.retained_groups)} / ${count(sampling.generated_groups)}`} note={olmo ? 'whole run · the gap is extra generation cost' : 'retained / generated'} />
      <SamplingMetric label="Sampling rounds" value={count(sampling.generation_rounds)} note="average per step · 1 means no refill was needed" />
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
  const [selectedWindow, setSelectedWindow] = useState<{ start: number; end: number } | null>(null);
  const steps = useMemo(() => [...sampling.steps].sort((left, right) => left.step - right.step), [sampling.steps]);
  if (!steps.length) return null;
  const classIds = [...new Set(steps.flatMap((step) => Object.keys(step.class_counts)))].sort();
  // Assign colors by identity, then use one full-run order for every stacked bar.
  // The visible range must not reorder segments or recolor a class.
  const colors = new Map(classIds.map((classId, index) => [classId, CLASS_COLORS[index % CLASS_COLORS.length]]));
  const classTotals = new Map(classIds.map((classId) => [
    classId,
    steps.reduce((total, step) => total + (step.class_counts[classId] ?? 0), 0),
  ]));
  const classes = [...classIds].sort((left, right) =>
    (classTotals.get(right) ?? 0) - (classTotals.get(left) ?? 0) || left.localeCompare(right));
  const latestStart = Math.max(0, steps.length - 20);
  const start = selectedWindow == null ? latestStart : Math.min(selectedWindow.start, steps.length - 1);
  const end = selectedWindow == null ? steps.length - 1 : Math.min(selectedWindow.end, steps.length - 1);
  const visibleSteps = steps.slice(start, end + 1);
  const sliderMaximum = Math.max(1, steps.length - 1);
  const chartMinimumWidth = Math.max(540, 96 + visibleSteps.length * 33);
  const isLatestWindow = start === latestStart && end === steps.length - 1;
  return <section className="obs-card mt-3 overflow-hidden" aria-label="Adaptive sampling distribution by step">
    <header className="flex flex-wrap items-end justify-between gap-3 border-b border-divider px-4 py-3">
      <div>
        <p className="type-eyebrow">CURRICULUM</p>
        <h2 className="mt-1 font-serif text-lg font-normal">What was sampled each step</h2>
      </div>
      <p className="max-w-xl text-[11px] leading-4 text-muted">Bars split each step's sampled groups by domain; the number above is groups sampled. The curriculum favours tasks likely to produce useful groups and spends some picks re-checking tasks whose evidence is thin or old. The table below shows what that cost and how new the picks were.</p>
    </header>
    <div className="px-4 pt-3">
      <div className="flex items-center justify-between gap-3 text-[11px]">
        <strong className="font-medium tabular-nums">Steps {steps[start].step}–{steps[end].step} of {steps.length}</strong>
        <button type="button" className="obs-control px-2 py-1 disabled:cursor-default disabled:opacity-60" disabled={isLatestWindow} onClick={() => setSelectedWindow(null)}>Latest 20</button>
      </div>
      <div className="relative mt-2 h-5" role="group" aria-label="Optimizer step range">
        <div className="absolute inset-x-0 top-2 h-1 rounded-full bg-subtle" />
        <div className="absolute top-2 h-1 rounded-full bg-accent/40" style={{ left: `${start / sliderMaximum * 100}%`, width: `${(end - start) / sliderMaximum * 100}%` }} />
        <input className="sampling-range" type="range" min={0} max={steps.length - 1} value={start} disabled={steps.length === 1} aria-label="First optimizer step" aria-valuetext={`Step ${steps[start].step}`} onChange={(event) => setSelectedWindow({ start: Math.min(Number(event.target.value), end), end })} />
        <input className="sampling-range" type="range" min={0} max={steps.length - 1} value={end} disabled={steps.length === 1} aria-label="Last optimizer step" aria-valuetext={`Step ${steps[end].step}`} onChange={(event) => setSelectedWindow({ start, end: Math.max(Number(event.target.value), start) })} />
      </div>
    </div>
    <div className="flex flex-wrap gap-x-4 gap-y-1 px-4 pb-1 pt-2">
      {classes.map((classId) => <span key={classId} className="inline-flex items-center gap-1.5 text-[10px] text-secondary"><span className="h-2 w-2 rounded-sm" style={{ backgroundColor: colors.get(classId) }} />{classId}</span>)}
    </div>
    <div className="overflow-x-auto px-4 pb-3">
      <div style={{ minWidth: chartMinimumWidth }}>
        <SamplingBarChart steps={visibleSteps} classes={classes} colors={colors} />
        <SamplingOutcomeMatrix steps={visibleSteps} />
      </div>
    </div>
  </section>;
}

function SamplingBarChart({ steps, classes, colors }: {
  steps: GRPOSamplingStep[];
  classes: string[];
  colors: Map<string, string>;
}) {
  const elementRef = useRef<HTMLDivElement>(null);
  const option = useMemo<echarts.EChartsCoreOption>(() => ({
    animation: false,
    grid: { left: 96, right: 8, top: 24, bottom: 23 },
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
    xAxis: { type: 'category', data: steps.map((step) => String(step.step)), axisTick: { show: false }, axisLine: { lineStyle: { color: '#d7d1d9' } }, axisLabel: { color: '#5f5a62', fontSize: 9, interval: 0 } },
    yAxis: { type: 'value', min: 0, max: 100, interval: 50, axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: '#817a83', fontSize: 9, formatter: '{value}%' }, splitLine: { lineStyle: { color: '#ebe7e1' } } },
    series: classes.map((classId) => ({
        name: classId,
        type: 'bar' as const,
        stack: 'classes',
        barCategoryGap: '2%',
        emphasis: { focus: 'series' },
        itemStyle: { color: colors.get(classId) },
        data: steps.map((step) => {
          const total = Object.values(step.class_counts).reduce((sum, value) => sum + value, 0);
          return total > 0 ? (step.class_counts[classId] ?? 0) / total * 100 : 0;
        }),
      })),
  }), [steps, classes, colors]);
  useEffect(() => {
    if (!elementRef.current) return;
    const chart = echarts.init(elementRef.current, undefined, { renderer: 'canvas' });
    chart.setOption(option);
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(elementRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);
  return <div className="relative">
    <div className="sr-only">
      {steps.map((step) => <p key={step.step}>Step {step.step}: {classes.map((classId) => `${classId} ${step.class_counts[classId] ?? 0}`).join(', ')}</p>)}
    </div>
    <div className="pointer-events-none absolute left-[96px] right-[8px] top-0 grid h-5 items-end text-center text-[9px] font-medium tabular-nums text-ink" style={{ gridTemplateColumns: `repeat(${steps.length}, minmax(0, 1fr))` }} aria-label="Candidate totals by step">
      {steps.map((step) => <span key={step.step}>{step.candidate_groups}</span>)}
    </div>
    <div ref={elementRef} className="h-[145px] w-full" role="img" aria-label="Stacked class distribution and total candidates by optimizer step" />
  </div>;
}

type OutcomeRow = { label: string; help: string; color: string; values: string[] };

function SamplingOutcomeMatrix({ steps }: { steps: GRPOSamplingStep[] }) {
  const kept = (step: GRPOSamplingStep) => step.retained_groups ?? null;
  const rows: OutcomeRow[] = [
    {
      label: 'Useful groups kept',
      help: 'Groups whose rollouts earned different rewards; these entered the update.',
      color: '#23878a',
      values: steps.map((step) => (kept(step) == null ? '—' : String(kept(step)))),
    },
    {
      label: 'Tied groups discarded',
      help: 'Groups where every rollout scored the same. They were generated but taught nothing, so they are extra cost.',
      color: '#a45179',
      values: steps.map((step) => (kept(step) == null ? '—' : String(Math.max(0, step.candidate_groups - (kept(step) ?? 0))))),
    },
    {
      label: 'Sampling rounds',
      help: 'Rounds of generation needed to collect enough useful groups. 1 means the first batch was enough.',
      color: '#817a83',
      values: steps.map((step) => String(step.refill_rounds || 1)),
    },
    {
      label: 'First-time tasks',
      help: 'Tasks sampled for the first time in this run.',
      color: '#3276a8',
      values: steps.map((step) => String(step.new_tasks)),
    },
    {
      label: 'Repeat tasks',
      help: 'Tasks sampled before: practice on tasks that produced useful groups, or re-checks of uncertain ones.',
      color: '#b96d32',
      values: steps.map((step) => String(Math.max(0, step.candidate_groups - step.new_tasks))),
    },
  ];
  // These only apply to the older quota curriculum; yield-first never reserves
  // novelty slots and refuses same-step repeats, so they stay hidden at zero.
  if (steps.some((step) => step.discovery_reserved > 0)) {
    rows.push({
      label: 'Novelty quota filled',
      help: 'Quota curriculum only: slots reserved for never-seen tasks, filled / reserved.',
      color: '#596579',
      values: steps.map((step) => `${step.discovery_fulfilled}/${step.discovery_reserved}`),
    });
  }
  if (steps.some((step) => step.duplicate_fallbacks > 0)) {
    rows.push({
      label: 'Same-step repeats',
      help: 'A task sampled twice in one step because distinct tasks ran out.',
      color: '#8a6c2f',
      values: steps.map((step) => String(step.duplicate_fallbacks)),
    });
  }
  return <div className="border-t border-divider pt-2">
    <h3 className="mb-1 text-[11px] font-medium">Sampling cost and task mix</h3>
    <table className="w-full table-fixed border-collapse text-[9px] tabular-nums" aria-label="Sampling cost and task mix by optimizer step">
      <colgroup><col style={{ width: 96 }} />{steps.map((step) => <col key={step.step} />)}</colgroup>
      <thead className="sr-only"><tr><th scope="col">Measure</th>{steps.map((step) => <th scope="col" key={step.step}>Step {step.step}</th>)}</tr></thead>
      <tbody>
        {rows.map((row) => <tr key={row.label} className="border-t border-divider/60">
          <th scope="row" title={row.help} className="cursor-help whitespace-nowrap py-1 text-left font-normal text-secondary underline decoration-divider decoration-dotted underline-offset-2">{row.label}</th>
          {row.values.map((value, index) => <td key={steps[index].step} className="px-0.5 py-1 text-center text-ink"><span className="inline-flex items-center justify-center gap-1"><span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ backgroundColor: row.color }} />{value}</span></td>)}
        </tr>)}
      </tbody>
    </table>
  </div>;
}
