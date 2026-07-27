import type { ServingCapacityWorkPackage } from '../../lib/api';

function value(number: number | null | undefined, digits = 1): string {
  return number == null ? '—' : number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function statusTone(state: string): string {
  if (state === 'eligible') return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  if (state === 'unsaturated' || state === 'insufficient_evidence' || state === 'incomparable') {
    return 'border-amber-200 bg-amber-50 text-amber-800';
  }
  return 'border-rose-200 bg-rose-50 text-rose-800';
}

export function ServingCapacityWorkPackageView({
  view,
  onOpenRun,
}: {
  view: ServingCapacityWorkPackage;
  onOpenRun: (runKey: string) => void;
}) {
  const plot = view.contenders.filter((contender) =>
    contender.comparable
    && contender.selected_point?.aggregate_output_tps != null
    && contender.selected_point?.p95_ttft_ms != null);
  const width = 720;
  const height = 250;
  const inset = { left: 64, right: 22, top: 22, bottom: 44 };
  const maxTps = Math.max(...plot.map((item) => item.selected_point?.aggregate_output_tps ?? 0), 1);
  const maxLatency = Math.max(...plot.map((item) => item.selected_point?.p95_ttft_ms ?? 0), 1);
  const x = (tps: number) => inset.left + (tps / maxTps) * (width - inset.left - inset.right);
  const y = (latency: number) => inset.top + (latency / maxLatency) * (height - inset.top - inset.bottom);
  return <section aria-labelledby="serving-capacity-work-package" className="mt-6 space-y-4">
    <div>
      <p className="type-eyebrow">SERVING CAPACITY</p>
      <h2 id="serving-capacity-work-package" className="mt-1 font-serif text-2xl">Comparable contenders</h2>
      <p className="mt-1 max-w-4xl text-xs leading-5 text-muted">{view.explanation}</p>
    </div>
    <section aria-label="Serving comparison basis" className="obs-card grid overflow-hidden sm:grid-cols-2 xl:grid-cols-5">
      {[
        ['Methodology', view.methodology === 'strict_pareto' ? 'Strict Pareto' : 'Historical compatibility'],
        ['Target', view.execution_target_id ?? 'Not established'],
        ['Workload', view.workload_id ?? 'Not established'],
        ['Comparable runs', String(view.contenders.filter((item) => item.comparable).length)],
        ['Pareto frontier', String(view.pareto.length)],
      ].map(([label, content]) => <div key={label} className="min-w-0 border-b border-r border-divider px-4 py-3"><span className="type-label">{label}</span><strong title={content} className="mt-1 block truncate text-[11px] font-medium">{content}</strong></div>)}
    </section>
    {plot.length > 0 && <figure className="obs-card overflow-hidden">
      <figcaption className="border-b border-divider px-4 py-3"><span className="type-eyebrow">PARETO VIEW</span><strong className="mt-1 block font-serif text-xl font-normal">Throughput versus p95 TTFT</strong><span className="mt-1 block text-[10px] text-muted">Higher throughput and lower latency are preferred. Green rings identify non-dominated eligible contenders; other results remain visible.</span></figcaption>
      <div className="p-4">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Serving contender throughput versus latency Pareto plot" className="h-auto w-full">
          <path d={`M${inset.left},${inset.top} V${height - inset.bottom} H${width - inset.right}`} fill="none" stroke="#a8a29e" />
          {plot.map((contender) => {
            const point = contender.selected_point!;
            const state = contender.eligibility.state;
            return <g key={contender.run_key}>
              <circle cx={x(point.aggregate_output_tps ?? 0)} cy={y(point.p95_ttft_ms ?? 0)} r={contender.pareto_member ? 9 : 6} fill={state === 'eligible' ? '#047857' : state === 'unsaturated' ? '#b45309' : '#be123c'} stroke={contender.pareto_member ? '#065f46' : 'white'} strokeWidth={contender.pareto_member ? 4 : 2}><title>{`${contender.display_name}: ${value(point.aggregate_output_tps)} TPS, ${value(point.p95_ttft_ms, 0)} ms p95 TTFT, ${contender.eligibility.label}`}</title></circle>
              <text x={x(point.aggregate_output_tps ?? 0) + 10} y={y(point.p95_ttft_ms ?? 0) - 8} fontSize="11" fill="#57534e">{contender.inference_binding_id?.replace(/^inference\//, '').split('@')[0] ?? contender.display_name}</text>
            </g>;
          })}
          <text x="8" y="18" fontSize="12" fill="#57534e">p95 TTFT</text>
          <text x={width - inset.right} y={height - 8} textAnchor="end" fontSize="12" fill="#57534e">Aggregate output TPS</text>
        </svg>
      </div>
    </figure>}
    <section className="obs-card overflow-hidden">
      <header className="border-b border-divider px-4 py-3"><p className="type-eyebrow">ALL EVIDENCE</p><h3 className="mt-1 font-serif text-xl">Contender decisions</h3></header>
      {!view.contenders.length ? <p className="p-6 text-center text-xs text-muted">No serving benchmark runs were recorded in this work package.</p> : <div className="overflow-x-auto">
        <table className="w-full min-w-[1080px] border-collapse text-left text-[10px]">
          <thead className="bg-subtle text-muted"><tr>{['Contender', 'Comparison', 'Best TPS', 'Concurrency', 'p95 TTFT', 'Peak VRAM', 'State', 'Frontier'].map((label) => <th key={label} className="border-b border-divider px-3 py-2 font-medium">{label}</th>)}</tr></thead>
          <tbody>{view.contenders.map((contender) => {
            const point = contender.selected_point;
            const state = contender.comparable ? contender.eligibility.state : 'incomparable';
            return <tr key={contender.run_key} className="bg-surface">
              <td className="border-b border-divider px-3 py-2.5"><button type="button" onClick={() => onOpenRun(contender.run_key)} className="max-w-[260px] text-left font-medium text-violet-700 hover:underline">{contender.inference_binding_id ?? contender.display_name}</button><span className="mt-1 block truncate text-[9px] text-muted">{contender.model_variant_id ?? 'Model not recorded'}</span></td>
              <td className="max-w-[260px] border-b border-divider px-3 py-2.5">{contender.comparable ? 'Comparable' : contender.comparability_reason}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{value(point?.aggregate_output_tps)}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{value(point?.concurrency, 0)}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point?.p95_ttft_ms == null ? '—' : `${value(point.p95_ttft_ms, 0)} ms`}</td>
              <td className="border-b border-divider px-3 py-2.5 font-mono">{point?.peak_vram_bytes == null ? '—' : `${(point.peak_vram_bytes / 1024 ** 3).toFixed(2)} GiB`}</td>
              <td className="border-b border-divider px-3 py-2.5"><span className={`inline-block border px-1.5 py-0.5 text-[8px] font-medium uppercase ${statusTone(state)}`}>{contender.comparable ? contender.eligibility.label : 'Incomparable'}</span></td>
              <td className="border-b border-divider px-3 py-2.5 font-medium">{contender.pareto_member ? 'Pareto member' : '—'}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>}
    </section>
  </section>;
}
