import { useEffect, useState } from 'react';

import {
  api,
  type ServingCapacityWorkPackage,
} from '../../lib/api';
import type {
  ExecutionTargetContext,
  RuntimeSettingGroup,
  ServingOperatingPoint,
  ServingRequirement,
  RunView,
} from '../../lib/api';

function number(value: number | null | undefined, digits = 1): string {
  return value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function measured(requirement: ServingRequirement): string {
  if (requirement.measured == null) return 'Not recorded';
  if (requirement.unit === 'ratio') return `${(requirement.measured * 100).toFixed(2)}%`;
  return `${number(requirement.measured, requirement.unit === 'tokens/s' ? 1 : 0)} ${requirement.unit}`;
}

function threshold(requirement: ServingRequirement): string {
  if (requirement.threshold == null) return 'Not configured';
  const comparator = requirement.operator === 'gte' ? '≥' : '≤';
  if (requirement.unit === 'ratio') return `${comparator} ${(requirement.threshold * 100).toFixed(2)}%`;
  return `${comparator} ${number(requirement.threshold, 0)} ${requirement.unit}`;
}

function statusTone(state: string): string {
  if (state === 'eligible' || state === 'pass' || state === 'valid') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  if (state === 'unsaturated' || state === 'unavailable' || state === 'incomplete') return 'border-amber-200 bg-amber-50 text-amber-800';
  return 'border-rose-200 bg-rose-50 text-rose-800';
}

function targetLabel(target: ExecutionTargetContext | undefined): string {
  if (!target) return 'Hardware not recorded';
  const memory = target.aggregate_memory_bytes == null
    ? null
    : `${(target.aggregate_memory_bytes / 1024 ** 3).toFixed(0)} GiB`;
  return [
    target.device_count ? `${target.device_count}×` : null,
    target.device_class,
    memory,
  ].filter(Boolean).join(' · ');
}

function pointCards(point: ServingOperatingPoint | undefined, tpsMargin: number | null | undefined) {
  return [
    ['Aggregate output TPS', number(point?.aggregate_output_tps), tpsMargin == null ? 'Constraint margin unavailable' : `${tpsMargin >= 0 ? '+' : ''}${number(tpsMargin)} tokens/s margin`],
    ['Operating concurrency', number(point?.concurrency, 0), 'Concurrent requests'],
    ['p95 TTFT', point?.p95_ttft_ms == null ? '—' : `${number(point.p95_ttft_ms, 0)} ms`, 'Time to first token'],
    ['p95 TPOT', point?.p95_tpot_ms == null ? '—' : `${number(point.p95_tpot_ms)} ms`, 'Per output token'],
    ['Mean response length', point?.output_tokens_mean == null ? '—' : `${number(point.output_tokens_mean, 0)} tokens`, point?.output_tokens_p95 == null ? 'p95 unavailable' : `p95 ${number(point.output_tokens_p95, 0)} tokens`],
    ['Measured requests', number(point?.attempted_requests, 0), `${point?.completed_requests ?? 0} completed`],
    ['Measurement duration', point?.measurement_seconds == null ? '—' : `${number(point.measurement_seconds)} s`, 'Timed inference window'],
    ['Peak VRAM', point?.peak_vram_bytes == null ? '—' : `${(point.peak_vram_bytes / 1024 ** 3).toFixed(2)} GiB`, 'Observed allocation'],
    ['Failure rate', point?.failure_rate == null ? '—' : `${(point.failure_rate * 100).toFixed(2)}%`, `${point?.failed_requests ?? 0}/${point?.attempted_requests ?? 0} requests`],
  ];
}

function compactBinding(value: string | null): string {
  if (!value) return 'Binding not recorded';
  return value.replace(/^inference\//, '').replace(/@\d+$/, '');
}

function CapacityTable({
  data,
  currentRunId,
}: {
  data: ServingCapacityWorkPackage | null;
  currentRunId: string;
}) {
  return <section aria-labelledby="capacity-points" className="obs-card mt-4 overflow-hidden">
    <header className="flex flex-wrap items-start justify-between gap-3 border-b border-divider px-4 py-3">
      <div><p className="type-eyebrow">CAPACITY EVIDENCE</p><h2 id="capacity-points" className="mt-1 font-serif text-xl">Concurrency points</h2></div>
      <p className="max-w-xl text-right text-[10px] leading-4 text-muted">{data?.explanation ?? 'Loading measured points…'}</p>
    </header>
    {!data ? <div className="px-4 py-8 text-center text-xs text-muted">Loading capacity points…</div> : !data.rows.length ? <div className="px-4 py-8 text-center text-xs text-muted">No serving-capacity points were recorded in this work package.</div> : <div className="overflow-x-auto">
      <table className="w-full min-w-[1120px] border-collapse text-left text-[10px]">
        <thead className="bg-subtle text-muted">
          <tr>{['Inference binding', 'Concurrency', 'Aggregate TPS', 'Mean output', 'p95 TTFT', 'p95 TPOT', 'Requests', 'Duration', 'Failures', 'Peak VRAM', 'Evidence state'].map((label) => <th key={label} className="whitespace-nowrap border-b border-divider px-3 py-2 font-medium">{label}</th>)}</tr>
        </thead>
        <tbody>
          {data.rows.map((row) => {
            const point = row.point;
            const current = row.locator.run_id === currentRunId;
            return <tr key={`${row.run_key}:${point.sweep_index}`} className={current ? 'bg-violet-50/60' : 'bg-surface'}>
              <td className="max-w-[260px] border-b border-divider px-3 py-2.5"><code title={row.inference_binding_id ?? undefined} className="block truncate">{compactBinding(row.inference_binding_id)}</code>{current && <span className="mt-1 inline-block text-[8px] font-medium uppercase tracking-[.12em] text-violet-700">Current run</span>}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.concurrency}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{number(point.aggregate_output_tps)}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.output_tokens_mean == null ? '—' : `${number(point.output_tokens_mean, 0)} tok`}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.p95_ttft_ms == null ? '—' : `${number(point.p95_ttft_ms, 0)} ms`}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.p95_tpot_ms == null ? '—' : `${number(point.p95_tpot_ms)} ms`}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.completed_requests}/{point.attempted_requests}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.measurement_seconds == null ? '—' : `${number(point.measurement_seconds)} s`}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.failed_requests}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point.peak_vram_bytes == null ? '—' : `${(point.peak_vram_bytes / 1024 ** 3).toFixed(2)} GiB`}</td>
              <td className="border-b border-divider px-3 py-2.5"><span className={`inline-block whitespace-nowrap border px-1.5 py-0.5 text-[8px] font-medium uppercase ${statusTone(row.point_state)}`}>{row.point_label}</span></td>
            </tr>;
          })}
        </tbody>
      </table>
    </div>}
  </section>;
}

function CapacityCurves({
  points,
  requirements,
}: {
  points: ServingOperatingPoint[];
  requirements: ServingRequirement[];
}) {
  const complete = points.filter((point) =>
    point.aggregate_output_tps != null && point.p95_ttft_ms != null);
  if (complete.length < 2) return null;
  const width = 720;
  const height = 230;
  const inset = { left: 58, right: 22, top: 22, bottom: 38 };
  const plotWidth = width - inset.left - inset.right;
  const plotHeight = height - inset.top - inset.bottom;
  const maxConcurrency = Math.max(...complete.map((point) => point.concurrency), 1);
  const maxTps = Math.max(...complete.map((point) => point.aggregate_output_tps ?? 0), 1);
  const maxTtft = Math.max(...complete.map((point) => point.p95_ttft_ms ?? 0), 1);
  const tpsLimit = requirements.find((item) => item.key === 'output_tps')?.threshold ?? null;
  const ttftLimit = requirements.find((item) => item.key === 'p95_ttft')?.threshold ?? null;
  const xConcurrency = (value: number) => inset.left + (value / maxConcurrency) * plotWidth;
  const xTps = (value: number) => inset.left + (value / Math.max(maxTps, tpsLimit ?? 0, 1)) * plotWidth;
  const yTps = (value: number) => inset.top + plotHeight - (value / Math.max(maxTps, tpsLimit ?? 0, 1)) * plotHeight;
  const yTtft = (value: number) => inset.top + (value / Math.max(maxTtft, ttftLimit ?? 0, 1)) * plotHeight;
  const capacityLine = complete.map((point) =>
    `${xConcurrency(point.concurrency)},${yTps(point.aggregate_output_tps ?? 0)}`).join(' ');
  return <section aria-labelledby="capacity-curves" className="obs-card mt-4 overflow-hidden">
    <header className="border-b border-divider px-4 py-3">
      <p className="type-eyebrow">OPERATING CURVES</p>
      <h2 id="capacity-curves" className="mt-1 font-serif text-xl">Throughput and latency under concurrency</h2>
      <p className="mt-1 text-[10px] leading-4 text-muted">The requirement lines come from the recorded project brief. Concurrency is the search dimension, not an acceptance threshold.</p>
    </header>
    <div className="grid gap-px bg-divider xl:grid-cols-2">
      <figure className="min-w-0 bg-surface p-4">
        <figcaption className="mb-2 text-[11px] font-medium">Aggregate output TPS by concurrency</figcaption>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Aggregate output tokens per second by concurrency" className="h-auto w-full">
          <path d={`M${inset.left},${inset.top} V${height - inset.bottom} H${width - inset.right}`} fill="none" stroke="#a8a29e" />
          {tpsLimit != null && <line x1={inset.left} x2={width - inset.right} y1={yTps(tpsLimit)} y2={yTps(tpsLimit)} stroke="#b45309" strokeDasharray="8 5" />}
          <polyline points={capacityLine} fill="none" stroke="#6d28d9" strokeWidth="3" />
          {complete.map((point) => <g key={point.sweep_index}>
            <circle cx={xConcurrency(point.concurrency)} cy={yTps(point.aggregate_output_tps ?? 0)} r="5" fill={point.valid ? '#047857' : '#be123c'} stroke="white" strokeWidth="2"><title>{`C${point.concurrency}: ${number(point.aggregate_output_tps)} TPS`}</title></circle>
            <text x={xConcurrency(point.concurrency)} y={height - 15} textAnchor="middle" fontSize="12" fill="#57534e">{point.concurrency}</text>
          </g>)}
          <text x="10" y="18" fontSize="12" fill="#57534e">TPS</text>
          <text x={width - inset.right} y={height - 4} textAnchor="end" fontSize="12" fill="#57534e">Concurrency</text>
          {tpsLimit != null && <text x={width - inset.right} y={Math.max(12, yTps(tpsLimit) - 6)} textAnchor="end" fontSize="11" fill="#92400e">{`${number(tpsLimit)} TPS minimum`}</text>}
        </svg>
      </figure>
      <figure className="min-w-0 bg-surface p-4">
        <figcaption className="mb-2 text-[11px] font-medium">Latency versus throughput</figcaption>
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="p95 time to first token versus aggregate output tokens per second" className="h-auto w-full">
          <path d={`M${inset.left},${inset.top} V${height - inset.bottom} H${width - inset.right}`} fill="none" stroke="#a8a29e" />
          {ttftLimit != null && <line x1={inset.left} x2={width - inset.right} y1={yTtft(ttftLimit)} y2={yTtft(ttftLimit)} stroke="#b45309" strokeDasharray="8 5" />}
          {complete.map((point) => <g key={point.sweep_index}>
            <circle cx={xTps(point.aggregate_output_tps ?? 0)} cy={yTtft(point.p95_ttft_ms ?? 0)} r="6" fill={point.valid ? '#047857' : '#be123c'} stroke="white" strokeWidth="2"><title>{`C${point.concurrency}: ${number(point.aggregate_output_tps)} TPS, ${number(point.p95_ttft_ms, 0)} ms p95 TTFT`}</title></circle>
            <text x={xTps(point.aggregate_output_tps ?? 0) + 8} y={yTtft(point.p95_ttft_ms ?? 0) - 7} fontSize="11" fill="#57534e">{`C${point.concurrency}`}</text>
          </g>)}
          <text x="10" y="18" fontSize="12" fill="#57534e">p95 TTFT</text>
          <text x={width - inset.right} y={height - 4} textAnchor="end" fontSize="12" fill="#57534e">Aggregate TPS</text>
          {ttftLimit != null && <text x={width - inset.right} y={Math.max(12, yTtft(ttftLimit) - 6)} textAnchor="end" fontSize="11" fill="#92400e">{`${number(ttftLimit, 0)} ms maximum`}</text>}
        </svg>
      </figure>
    </div>
    <ul className="sr-only">{complete.map((point) => <li key={point.sweep_index}>{`Concurrency ${point.concurrency}, ${number(point.aggregate_output_tps)} aggregate TPS, ${number(point.p95_ttft_ms, 0)} milliseconds p95 TTFT, ${point.valid ? 'valid' : 'constraint failed'}.`}</li>)}</ul>
  </section>;
}

function RuntimeSettings({ groups }: { groups: RuntimeSettingGroup[] }) {
  return <section aria-labelledby="runtime-settings" className="obs-card overflow-hidden">
    <header className="flex flex-wrap items-start justify-between gap-3 border-b border-divider px-4 py-3">
      <div><p className="type-eyebrow">BACKEND RUNTIME</p><h2 id="runtime-settings" className="mt-1 font-serif text-xl">Serving configuration</h2></div>
      <span className="text-[10px] text-muted">Curated here · complete binding in Run config</span>
    </header>
    <div className="grid gap-px bg-divider md:grid-cols-2">
      {groups.map((group) => <article key={group.key} className="bg-surface px-4 py-3">
        <h3 className="text-[11px] font-medium">{group.label}</h3>
        <dl className="mt-2 space-y-1.5">
          {group.settings.map((setting) => <div key={setting.key} className="flex items-start justify-between gap-4 text-[10px]">
            <dt className="text-muted">{setting.label}</dt>
            <dd className="max-w-[60%] break-words text-right font-mono text-secondary">
              {setting.state === 'redacted' ? 'Redacted' : typeof setting.value === 'object' ? JSON.stringify(setting.value) : String(setting.value)}
              {setting.unit ? ` ${setting.unit}` : ''}
            </dd>
          </div>)}
        </dl>
      </article>)}
    </div>
  </section>;
}

export function ServingBenchmarkOverview({
  response,
  onRunConfig,
  sourceId,
}: {
  response: RunView;
  onRunConfig: () => void;
  sourceId: string;
}) {
  const view = response.view;
  const eligibility = view.eligibility;
  const requirements = view.requirements ?? [];
  const point = view.selected_point ?? view.operating_points?.[0];
  const target = view.execution_targets?.[0];
  const tpsRequirement = requirements.find((requirement) => requirement.key === 'output_tps');
  const population = view.population;
  const [capacity, setCapacity] = useState<ServingCapacityWorkPackage | null>(null);
  const inRunCapacity: ServingCapacityWorkPackage | null = (view.operating_points?.length ?? 0) > 1
    ? {
        schema_version: 1,
        project_id: view.run.project_id,
        work_package_id: view.run.work_package_id,
        methodology: 'single_run_sweep',
        explanation: 'All points below were measured in this run with one model load and one resolved serving binding.',
        requirements,
        execution_target_id: view.execution_target_id ?? null,
        workload_id: view.workload_id ?? null,
        corpus_digest: population?.corpus_digest ?? null,
        requirements_digest: eligibility?.requirements_digest ?? null,
        calculator_version: eligibility?.calculator_version ?? null,
        contenders: [],
        pareto: [],
        rows: (view.operating_points ?? []).map((operatingPoint) => ({
          locator: { source_id: sourceId, run_id: view.run.run_id },
          run_key: `${sourceId}:${view.run.run_id}`,
          display_name: view.run.display_name,
          started_at: view.run.started_at,
          model_variant_id: view.model_variant_id ?? null,
          inference_binding_id: view.inference_binding_id ?? null,
          inference_backend: view.inference_backend ?? null,
          workload_id: view.workload_id ?? null,
          execution_target_id: view.execution_target_id ?? null,
          requirements_digest: eligibility?.requirements_digest ?? null,
          point: operatingPoint,
          point_state: operatingPoint.evidence_state !== 'complete' ? 'incomplete' : operatingPoint.valid ? 'valid' : 'constraint_failed',
          point_label: operatingPoint.terminal_status
            ? `${operatingPoint.terminal_status.replaceAll('_', ' ')} boundary`
            : operatingPoint.evidence_state !== 'complete'
              ? 'Incomplete evidence'
              : operatingPoint.valid
                ? 'Valid point'
                : 'Constraint missed',
          eligibility: eligibility!,
        })),
      }
    : null;
  useEffect(() => {
    if ((view.operating_points?.length ?? 0) > 1) {
      setCapacity(null);
      return;
    }
    let active = true;
    setCapacity(null);
    void api.servingCapacity(view.run.work_package_id, view.run.project_id, sourceId)
      .then((value) => { if (active) setCapacity(value); })
      .catch(() => { if (active) setCapacity({ schema_version: 1, project_id: view.run.project_id, work_package_id: view.run.work_package_id, methodology: 'cross_run_compatibility', explanation: 'Capacity-point evidence is currently unavailable.', requirements: [], execution_target_id: null, workload_id: null, corpus_digest: null, requirements_digest: null, calculator_version: null, contenders: [], pareto: [], rows: [] }); });
    return () => { active = false; };
  }, [sourceId, view.operating_points?.length, view.run.project_id, view.run.work_package_id]);
  if (!eligibility) return null;

  return <>
    <div>
      <p className="type-eyebrow">SERVING CAPACITY</p>
      <h1 className="type-page-title mt-1.5">Serving benchmark</h1>
      <p className="type-page-subtitle mt-2">{view.question}</p>
    </div>

    <section aria-label="Serving benchmark decision" className={`mt-5 border px-4 py-3 ${statusTone(eligibility.state)}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-[10px] font-medium uppercase tracking-[.13em]">Constraint-relative result</p>
          <h2 className="mt-1 font-serif text-2xl">{eligibility.label}</h2>
          <p className="mt-1 max-w-3xl text-[11px] leading-5">{eligibility.reason}</p>
        </div>
        <div className="text-right text-[9px] leading-4 opacity-80">
          <code>{eligibility.calculator_version}</code>
          <div>{eligibility.saturation_state} sweep</div>
        </div>
      </div>
    </section>

    <section aria-label="Benchmark identity" className="obs-card mt-4 grid overflow-hidden sm:grid-cols-2 xl:grid-cols-5">
      {[
        ['Model', view.model_variant_id ?? 'Not recorded'],
        ['Backend', view.inference_backend ?? 'Not recorded'],
        ['Workload', view.workload_id ?? 'Not recorded'],
        ['Hardware profile', targetLabel(target)],
        ['Context allocation', point?.context_tokens == null ? 'Not recorded' : `${number(point.context_tokens, 0)} tokens`],
      ].map(([label, value]) => <div key={label} className="min-w-0 border-b border-r border-divider px-4 py-3">
        <span className="type-label">{label}</span>
        <strong title={value} className="mt-1.5 block truncate text-[11px] font-medium">{value}</strong>
      </div>)}
    </section>

    <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <section aria-labelledby="measured-point" className="obs-card overflow-hidden">
        <header className="border-b border-divider px-4 py-3">
          <p className="type-eyebrow">MEASURED OPERATING POINT</p>
          <h2 id="measured-point" className="mt-1 font-serif text-xl">
            {eligibility.state === 'unsaturated' ? 'Best observed point; not final capacity' : 'Selected valid capacity point'}
          </h2>
        </header>
        <div className="grid grid-cols-2 lg:grid-cols-3">
          {pointCards(point, tpsRequirement?.margin).map(([label, value, detail]) => <div key={label} className="border-b border-r border-divider px-4 py-4">
            <p className="type-label">{label}</p>
            <strong className="mt-1 block font-serif text-2xl font-normal">{value}</strong>
            <p className="mt-1 text-[9px] text-muted">{detail}</p>
          </div>)}
        </div>
      </section>

      <section aria-labelledby="product-constraints" className="obs-card overflow-hidden">
        <header className="border-b border-divider px-4 py-3">
          <p className="type-eyebrow">PROJECT POLICY</p>
          <h2 id="product-constraints" className="mt-1 font-serif text-xl">Product constraints</h2>
        </header>
        <div>
          {requirements.map((requirement) => <article key={requirement.key} className="border-b border-divider px-4 py-3 last:border-b-0" title={requirement.explanation}>
            <div className="flex items-start justify-between gap-3">
              <div><h3 className="text-[11px] font-medium">{requirement.label}</h3><p className="mt-1 text-[9px] text-muted">{threshold(requirement)}</p></div>
              <span className={`border px-1.5 py-0.5 text-[9px] font-medium uppercase ${statusTone(requirement.state)}`}>{requirement.state}</span>
            </div>
            <p className="mt-2 font-mono text-[10px] text-secondary">{measured(requirement)}</p>
          </article>)}
        </div>
      </section>
    </div>

    <CapacityCurves points={view.operating_points ?? []} requirements={requirements} />
    <CapacityTable data={inRunCapacity ?? capacity} currentRunId={view.run.run_id} />

    <div className="mt-4 grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
      <RuntimeSettings groups={view.runtime_settings ?? []} />
      <section aria-labelledby="benchmark-population" className="obs-card overflow-hidden">
        <header className="border-b border-divider px-4 py-3"><p className="type-eyebrow">BENCHMARK POPULATION</p><h2 id="benchmark-population" className="mt-1 font-serif text-xl">Representative requests</h2></header>
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-[10px] text-amber-800">Capacity only; task correctness was not scored.</div>
        {population?.output_length_policy === 'fixed' && <div className="border-b border-sky-200 bg-sky-50 px-4 py-2 text-[10px] leading-4 text-sky-900">
          Fixed-length systems run: generation continues to the configured output target, so response length describes decode work rather than natural stopping behavior.
        </div>}
        <dl className="grid grid-cols-2 gap-px bg-divider">
          {[
            ['Corpus', population?.corpus_id ?? 'Not recorded'],
            ['Revision', population?.corpus_revision ?? 'Not recorded'],
            ['Cohort', population?.cohort ?? 'Not recorded'],
            ['Measured records', number(population?.measured_records, 0)],
            ['Mean input', population?.input_tokens_mean == null ? '—' : `${number(population.input_tokens_mean, 0)} tokens`],
            ['p95 input', population?.input_tokens_p95 == null ? '—' : `${number(population.input_tokens_p95, 0)} tokens`],
            ['Fixed output target', population?.output_token_budget == null ? '—' : `${number(population.output_token_budget, 0)} tokens`],
            ['Output target hit', population?.output_target_hit_rate == null ? '—' : `${(population.output_target_hit_rate * 100).toFixed(1)}%`],
            ['Mean response', point?.output_tokens_mean == null ? '—' : `${number(point.output_tokens_mean, 0)} tokens`],
            ['p95 response', point?.output_tokens_p95 == null ? '—' : `${number(point.output_tokens_p95, 0)} tokens`],
          ].map(([label, value]) => <div key={label} className="min-w-0 bg-surface px-3 py-2.5"><dt className="type-label">{label}</dt><dd title={value} className="mt-1 truncate text-[10px]">{value}</dd></div>)}
        </dl>
        <button type="button" onClick={onRunConfig} className="w-full border-t border-divider px-4 py-3 text-left text-[10px] font-medium text-violet-700 hover:bg-subtle">Open complete resolved binding →</button>
      </section>
    </div>
  </>;
}
