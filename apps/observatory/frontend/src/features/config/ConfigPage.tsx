import { useEffect, useMemo, useState, type ReactNode } from 'react';
import {
  ArrowRight,
  Check,
  Copy,
  Info,
  Lightbulb,
  MagnifyingGlass,
  SlidersHorizontal,
  Warning,
  WarningOctagon,
  X,
} from '@phosphor-icons/react';
import type {
  ConfigurationFinding,
  ConfigurationReview,
  SettingsRecommendation,
  StepCalibration,
  StepCapacityView,
} from '../../lib/api';

export type ConfigGroup = { title: string; description: string; keys: string[] };

type Severity = ConfigurationFinding['severity'];
type Row = { path: string; key: string; value: unknown; missing?: boolean; suggested?: unknown };
type Section = { path: string; title: string; rows: Row[] };
type Card = {
  role: string;
  label: string;
  id: string;
  revision: string | null;
  layer: string | null;
  sections: Section[];
};

const SEVERITY_ORDER: Severity[] = ['error', 'warning', 'recommendation', 'info'];

const SEVERITY: Record<Severity, { label: string; plural: string; icon: ReactNode; text: string; soft: string; tint: string; edge: string; dot: string }> = {
  error: {
    label: 'Error',
    plural: 'errors',
    icon: <WarningOctagon size={14} weight="fill" aria-hidden="true" />,
    text: 'text-rose-700',
    soft: 'bg-rose-50',
    tint: 'bg-rose-50/60',
    edge: 'border-l-rose-500',
    dot: 'bg-rose-500',
  },
  warning: {
    label: 'Warning',
    plural: 'warnings',
    icon: <Warning size={14} weight="fill" aria-hidden="true" />,
    text: 'text-amber-800',
    soft: 'bg-amber-50',
    tint: 'bg-amber-50/60',
    edge: 'border-l-amber-500',
    dot: 'bg-amber-500',
  },
  recommendation: {
    label: 'Recommendation',
    plural: 'recommendations',
    icon: <Lightbulb size={14} weight="fill" aria-hidden="true" />,
    text: 'text-violet-700',
    soft: 'bg-violet-50',
    tint: 'bg-violet-50/50',
    edge: 'border-l-violet-500',
    dot: 'bg-violet-500',
  },
  info: {
    label: 'Acknowledged',
    plural: 'acknowledged',
    icon: <Info size={14} weight="fill" aria-hidden="true" />,
    text: 'text-secondary',
    soft: 'bg-subtle',
    tint: 'bg-subtle/60',
    edge: 'border-l-divider',
    dot: 'bg-muted',
  },
};

const SKIP_FIELDS = new Set(['selection_id', 'id', 'revision', 'source_layer', 'overlay_id', 'ref']);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function humanize(value: string): string {
  return value.replaceAll('_', ' ').replace(/^\w/, (letter) => letter.toUpperCase());
}

export function formatValue(value: unknown): string {
  if (value === undefined || value === null) return '—';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (value !== 0 && Math.abs(value) < 0.001) return value.toExponential(1);
    if (Number.isInteger(value) && Math.abs(value) >= 1024 ** 3) return `${value.toLocaleString()} (${(value / 1024 ** 3).toFixed(2)} GiB)`;
    return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
  }
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
}

function flatten(value: Record<string, unknown>, prefix: string, rows: Row[]) {
  for (const [key, field] of Object.entries(value)) {
    const path = `${prefix}.${key}`;
    if (isRecord(field) && Object.keys(field).length > 0) flatten(field, path, rows);
    else rows.push({ path, key: path.split('.').slice(1).join('.'), value: field });
  }
}

function cardFor(role: string, raw: unknown, labels: Record<string, string>): Card | null {
  if (!isRecord(raw)) return null;
  const resolved = isRecord(raw.resolved) ? raw.resolved : Object.fromEntries(Object.entries(raw).filter(([key]) => !SKIP_FIELDS.has(key)));
  const general: Row[] = [];
  const sections: Section[] = [];
  for (const [key, field] of Object.entries(resolved)) {
    const path = `${role}.${key}`;
    if (isRecord(field) && Object.keys(field).length > 0) {
      const rows: Row[] = [];
      flatten(field, path, rows);
      sections.push({ path, title: humanize(key), rows: rows.map((row) => ({ ...row, key: row.path.slice(path.length + 1) })) });
    } else {
      general.push({ path, key, value: field });
    }
  }
  const id = raw.selection_id ?? raw.id ?? resolved.id ?? resolved.work_package_id;
  return {
    role,
    label: labels[role] ?? humanize(role),
    id: typeof id === 'string' ? id : role,
    revision: typeof raw.revision === 'string' ? raw.revision : null,
    layer: typeof raw.source_layer === 'string' ? raw.source_layer : null,
    sections: [...(general.length ? [{ path: role, title: 'Settings', rows: general }] : []), ...sections],
  };
}

function severityCounts(findings: ConfigurationFinding[]) {
  const counts: Record<Severity, number> = { error: 0, warning: 0, recommendation: 0, info: 0 };
  for (const finding of findings) counts[finding.severity] += 1;
  return counts;
}

function worst(findings: ConfigurationFinding[]): Severity | null {
  return SEVERITY_ORDER.find((severity) => findings.some((finding) => finding.severity === severity)) ?? null;
}

function scrollToPath(path: string) {
  const element = document.getElementById(`cfg-${path}`);
  if (!element) return;
  element.scrollIntoView({ behavior: 'smooth', block: 'center' });
  element.classList.add('ring-2', 'ring-violet-400');
  window.setTimeout(() => element.classList.remove('ring-2', 'ring-violet-400'), 1600);
}

function SeverityBadge({ severity, compact = false }: { severity: Severity; compact?: boolean }) {
  const style = SEVERITY[severity];
  return <span className={`inline-flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium ${style.soft} ${style.text}`}>{style.icon}{!compact && style.label}</span>;
}

function FindingItem({ finding, anchored }: { finding: ConfigurationFinding; anchored: boolean }) {
  const style = SEVERITY[finding.severity];
  return <li className={`border-l-2 ${style.edge} px-4 py-3`}>
    <div className="flex flex-wrap items-center gap-2">
      <SeverityBadge severity={finding.severity} />
      <code className="font-mono text-[10px] text-secondary">{finding.code}</code>
      <span className="rounded border border-divider px-1.5 py-0.5 text-[9px] uppercase tracking-wide text-muted">{finding.source === 'calculator' ? 'Calculator' : 'Rule'}</span>
      {anchored
        ? <button type="button" onClick={() => scrollToPath(finding.path)} className="ml-auto inline-flex items-center gap-1 font-mono text-[10px] text-violet-700 hover:underline">{finding.path}<ArrowRight size={11} aria-hidden="true" /></button>
        : <code className="ml-auto font-mono text-[10px] text-muted">{finding.path}</code>}
    </div>
    <p className="mt-1.5 text-[12px] leading-5 text-ink">{finding.message}</p>
    {finding.hint && <p className="mt-1 text-[11px] leading-5 text-secondary"><span className="font-medium text-ink">Fix: </span>{finding.hint}</p>}
  </li>;
}

function ReviewSummary({ review, anchoredPaths, onRecommend }: { review: ConfigurationReview; anchoredPaths: Set<string>; onRecommend: (() => void) | null }) {
  const [expanded, setExpanded] = useState<Severity | 'all'>('all');
  const all = review.findings ?? [];
  const counts = severityCounts(all);
  const ordered = [...all].sort((left, right) => SEVERITY_ORDER.indexOf(left.severity) - SEVERITY_ORDER.indexOf(right.severity));
  const visible = expanded === 'all' ? ordered : ordered.filter((finding) => finding.severity === expanded);
  return <section aria-labelledby="config-review" className="obs-card mt-5 overflow-hidden">
    <header className="flex flex-wrap items-center gap-3 border-b border-divider px-4 py-3">
      <div className="mr-auto">
        <h2 id="config-review" className="text-sm font-semibold text-ink">Configuration review</h2>
        <p className="mt-0.5 text-[11px] text-muted">Performance and LoRA rules plus the settings calculator, evaluated on this run’s recorded selections.</p>
      </div>
      <div role="tablist" aria-label="Filter findings" className="flex flex-wrap gap-1.5">
        <button role="tab" aria-selected={expanded === 'all'} onClick={() => setExpanded('all')} className={`rounded-full border px-2.5 py-1 text-[11px] ${expanded === 'all' ? 'border-ink bg-ink text-white' : 'border-divider text-secondary hover:bg-subtle'}`}>All {all.length}</button>
        {SEVERITY_ORDER.filter((severity) => counts[severity] > 0).map((severity) => <button key={severity} role="tab" aria-selected={expanded === severity} onClick={() => setExpanded(severity)} className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] ${expanded === severity ? 'border-ink bg-ink text-white' : 'border-divider text-secondary hover:bg-subtle'}`}><span className={`h-1.5 w-1.5 rounded-full ${SEVERITY[severity].dot}`} />{counts[severity]} {SEVERITY[severity].plural}</button>)}
      </div>
      {onRecommend && <button type="button" onClick={onRecommend} className="inline-flex items-center gap-1.5 rounded-md bg-violet-700 px-3 py-1.5 text-[11px] font-medium text-white hover:bg-violet-800"><SlidersHorizontal size={14} aria-hidden="true" />Recommended settings</button>}
    </header>
    {all.length === 0
      ? <p className="flex items-center gap-2 px-4 py-4 text-[12px] text-emerald-700"><Check size={15} weight="bold" aria-hidden="true" />No findings: the recorded settings follow the measured defaults and reference recipes.</p>
      : <ul className="divide-y divide-divider">{visible.map((finding, index) => <FindingItem key={`${finding.code}:${finding.path}:${index}`} finding={finding} anchored={anchoredPaths.has(finding.path)} />)}</ul>}
    {review.calculator === 'disabled' && <p className="border-t border-divider bg-subtle px-4 py-2 text-[10px] text-muted">The settings calculator is disabled on this Observatory; only rule findings are shown.</p>}
  </section>;
}

function seconds(value: number | null | undefined): string {
  if (value == null) return '—';
  if (value >= 3600) return `${(value / 3600).toFixed(1)} h`;
  if (value >= 120) return `${(value / 60).toFixed(1)} min`;
  return `${value.toFixed(0)} s`;
}

export function StepSizing({ step: raw, calibration, role }: { step: StepCapacityView; calibration: StepCalibration | null | undefined; role: string }) {
  const step = { ...raw, options: raw.options ?? [] };
  const calibrated = step.options.some((option) => option.estimated_step_seconds != null);
  return <section aria-labelledby={`step-${role}`} className="obs-card mt-5 overflow-hidden">
    <header className="border-b border-divider px-4 py-3">
      <h2 id={`step-${role}`} className="text-sm font-semibold text-ink">Step sizing</h2>
      <p className="mt-0.5 text-[11px] leading-5 text-muted">{step.reason}. Oversampling takes at most 20% of a step; the rest of the margin goes to prompts.</p>
    </header>
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-left text-[11px]">
        <thead className="bg-subtle text-[10px] uppercase tracking-wide text-muted">
          <tr>
            <th className="px-4 py-2 font-medium">Option</th>
            <th className="px-3 py-2 text-right font-medium">Prompts / step</th>
            <th className="px-3 py-2 text-right font-medium">Oversampled groups</th>
            <th className="px-3 py-2 text-right font-medium">Rows</th>
            <th className="px-3 py-2 text-right font-medium">Waves</th>
            {calibrated && <th className="px-3 py-2 text-right font-medium">Est. rollout / step</th>}
            <th className="px-3 py-2 text-right font-medium">{calibrated ? 'Est. step time' : 'Step time'}</th>
            <th className="px-3 py-2 text-right font-medium">Rows / s</th>
            <th className="px-4 py-2 text-right font-medium">Decode-only rows / s</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-divider">
          {step.options.map((option) => {
            const current = option.label === 'current';
            const recommended = option.label === 'recommended';
            return <tr key={option.label} className={recommended ? 'bg-violet-50/60' : current ? 'bg-surface' : ''}>
              <td className="px-4 py-2"><span className={`font-medium capitalize ${recommended ? 'text-violet-700' : 'text-ink'}`}>{option.label}</span>{current && <span className="ml-1.5 text-[10px] text-muted">this run</span>}</td>
              <td className="px-3 py-2 text-right font-mono">{option.prompts_per_step}</td>
              <td className="px-3 py-2 text-right font-mono">{option.oversample_groups || '—'}</td>
              <td className="px-3 py-2 text-right font-mono">{option.rows}</td>
              <td className="px-3 py-2 text-right font-mono">{option.waves}</td>
              {calibrated && <td className="px-3 py-2 text-right font-mono">{seconds(option.estimated_rollout_seconds)}</td>}
              <td className="px-3 py-2 text-right font-mono">{calibrated ? seconds(option.estimated_step_seconds) : `${option.relative_step_time.toFixed(2)}×`}</td>
              <td className="px-3 py-2 text-right font-mono">{option.estimated_relative_rows_per_second != null ? `${option.estimated_relative_rows_per_second.toFixed(2)}×` : '—'}</td>
              <td className="px-4 py-2 text-right font-mono text-muted">{option.relative_rows_per_second.toFixed(2)}×</td>
            </tr>;
          })}
        </tbody>
      </table>
    </div>
    <p className="border-t border-divider px-4 py-2 text-[10px] leading-4 text-muted">
      {calibration
        ? `Scaled from this run’s measured ${calibration.rounds_per_step.toFixed(1)} rollout rounds of ${seconds(calibration.rollout_seconds)}${calibration.step_seconds ? ` and ${seconds(calibration.step_seconds)} steps` : ''} (mean of ${calibration.steps} steps). Within a round only decoding slows with more parallel episodes; tool calls, prefill and the slowest episode do not. The LoRA update and reward scoring scale with rows, so compare rows per second, not step time.`
        : 'Relative to this run’s step, from a memory-bandwidth decode estimate; no measured step times were recorded.'}
      {' '}{step.fits} typical-length sequences fit in memory; throughput flattens past {step.useful}.
    </p>
  </section>;
}

function ValueCell({ row }: { row: Row }) {
  const value = row.value;
  if (row.missing) return <span className="text-[11px] italic text-muted">not recorded</span>;
  if (typeof value === 'boolean') return <span className={`inline-flex rounded px-1.5 py-0.5 font-mono text-[10px] ${value ? 'bg-emerald-50 text-emerald-700' : 'bg-subtle text-secondary'}`}>{String(value)}</span>;
  if (Array.isArray(value)) {
    if (value.every((item) => !isRecord(item) && !Array.isArray(item))) {
      return <span className="flex flex-wrap gap-1">{value.length === 0 ? <span className="text-muted">[]</span> : value.map((item, index) => <span key={index} className="rounded bg-subtle px-1.5 py-0.5 font-mono text-[10px] text-secondary">{String(item)}</span>)}</span>;
    }
    return <code className="break-all font-mono text-[10px] text-secondary">{JSON.stringify(value)}</code>;
  }
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  return <code title={typeof value === 'string' ? value : undefined} className="break-all font-mono text-[11px] text-ink">{formatValue(value)}</code>;
}

function ConfigRow({ row, findings }: { row: Row; findings: ConfigurationFinding[] }) {
  const severity = worst(findings);
  const style = severity ? SEVERITY[severity] : null;
  const changed = row.suggested !== undefined;
  return <div id={`cfg-${row.path}`} className={`scroll-mt-24 rounded-sm border-l-2 transition-shadow ${style ? `${style.edge} ${style.tint}` : 'border-l-transparent'}`}>
    <div className="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)] items-start gap-3 px-3 py-1.5">
      <code className="break-all pt-0.5 font-mono text-[11px] text-secondary">{row.key}</code>
      <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <ValueCell row={row} />
        {changed && <span className="inline-flex items-center gap-1 rounded bg-violet-50 px-1.5 py-0.5 font-mono text-[10px] text-violet-700" title="Settings calculator suggestion"><ArrowRight size={10} aria-hidden="true" />{formatValue(row.suggested)}</span>}
        {findings.map((finding) => <span key={finding.code} className={`inline-flex items-center gap-1 text-[10px] ${SEVERITY[finding.severity].text}`}>{SEVERITY[finding.severity].icon}<code className="font-mono">{finding.code}</code></span>)}
      </div>
    </div>
    {findings.length > 0 && <div className="px-3 pb-2 pl-[calc(40%+0.75rem)] text-[11px] leading-5 text-secondary">{findings.map((finding) => <p key={finding.code}>{finding.message}{finding.hint ? <span className="text-ink"> {finding.hint}</span> : null}</p>)}</div>}
  </div>;
}

function ConfigCard({ card, findingsByPath, filter, flaggedOnly }: { card: Card; findingsByPath: Map<string, ConfigurationFinding[]>; filter: string; flaggedOnly: boolean }) {
  const cardFindings = [...findingsByPath.entries()].filter(([path]) => path === card.role || path.startsWith(`${card.role}.`)).flatMap(([, items]) => items);
  const severity = worst(cardFindings);
  const sections = card.sections.map((section) => ({
    ...section,
    rows: section.rows.filter((row) => (!filter || row.path.toLowerCase().includes(filter) || formatValue(row.value).toLowerCase().includes(filter))
      && (!flaggedOnly || (findingsByPath.get(row.path)?.length ?? 0) > 0 || row.suggested !== undefined)),
  })).filter((section) => section.rows.length > 0 || (findingsByPath.get(section.path)?.length ?? 0) > 0);
  if ((filter || flaggedOnly) && sections.length === 0) return null;
  return <article id={`cfg-${card.role}`} aria-label={`${card.label} selection`} className="obs-card scroll-mt-24 overflow-hidden">
    <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-divider bg-subtle/60 px-4 py-2.5">
      <h3 className="text-[12px] font-semibold text-ink">{card.label}</h3>
      <code title={card.id} className="min-w-0 truncate font-mono text-[11px] text-secondary">{card.id}{card.revision ? `@${card.revision}` : ''}</code>
      {card.layer && <span className="rounded border border-divider bg-surface px-1.5 py-0.5 text-[9px] text-muted">{card.layer}</span>}
      {severity && <span className="ml-auto"><SeverityBadge severity={severity} />{cardFindings.length > 1 && <span className="ml-1 text-[10px] text-muted">{cardFindings.length} findings</span>}</span>}
    </header>
    <div className="divide-y divide-divider">
      {sections.map((section) => {
        const sectionFindings = section.path === card.role ? [] : findingsByPath.get(section.path) ?? [];
        return <div key={section.path} id={section.path === card.role ? undefined : `cfg-${section.path}`} className="scroll-mt-24 py-1.5">
          <div className="flex items-center gap-2 px-3 pb-0.5 pt-1"><span className="type-label">{section.title}</span>{sectionFindings.map((finding) => <SeverityBadge key={finding.code} severity={finding.severity} compact />)}</div>
          {sectionFindings.length > 0 && <div className="mx-3 mb-1 rounded bg-subtle px-2 py-1.5 text-[11px] leading-5 text-secondary">{sectionFindings.map((finding) => <p key={finding.code}><code className="font-mono text-[10px]">{finding.code}</code>: {finding.message}</p>)}</div>}
          {section.rows.map((row) => <ConfigRow key={row.path} row={row} findings={findingsByPath.get(row.path) ?? []} />)}
        </div>;
      })}
    </div>
  </article>;
}

function yamlValue(value: unknown): string {
  if (typeof value === 'string') return /^[\w./@-]+$/.test(value) ? value : JSON.stringify(value);
  return JSON.stringify(value);
}

function MemoryBar({ memory }: { memory: Record<string, number> }) {
  const parts: Array<[string, string]> = [['weights', 'bg-violet-600'], ['draft_weights', 'bg-violet-300'], ['kv_cache', 'bg-sky-500'], ['cuda_graphs_estimate', 'bg-amber-400'], ['workspace_estimate', 'bg-orange-300'], ['trainer', 'bg-rose-400'], ['other_engines', 'bg-slate-400'], ['free', 'bg-divider']];
  const total = parts.reduce((sum, [key]) => sum + Math.max(memory[key] ?? 0, 0), 0) || 1;
  return <div>
    <div className="flex h-2.5 overflow-hidden rounded-full bg-subtle">{parts.map(([key, color]) => (memory[key] ?? 0) > 0 && <span key={key} className={color} style={{ width: `${((memory[key] ?? 0) / total) * 100}%` }} title={`${humanize(key)}: ${memory[key]} GB`} />)}</div>
    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-secondary">{parts.map(([key, color]) => memory[key] !== undefined && <span key={key} className="inline-flex items-center gap-1"><span className={`h-2 w-2 rounded-sm ${color}`} />{humanize(key.replace('_estimate', ''))} {memory[key]} GB</span>)}</div>
  </div>;
}

function RecommendationPanel({ recommendations, onClose }: { recommendations: SettingsRecommendation[]; onClose: () => void }) {
  const [active, setActive] = useState(0);
  const [changesOnly, setChangesOnly] = useState(true);
  const [copied, setCopied] = useState(false);
  const selected = recommendations[Math.min(active, recommendations.length - 1)];
  const recommendation = { ...selected, settings: selected.settings ?? [], environment: selected.environment ?? {}, memory_gb: selected.memory_gb ?? {}, notes: selected.notes ?? [] };
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  const rows = recommendation.settings.filter((row) => !changesOnly || row.changed);
  const yaml = `engine:\n${recommendation.settings.map((row) => `  ${row.key}: ${yamlValue(row.suggested)}`).join('\n')}${Object.keys(recommendation.environment).length ? `\nenvironment:\n${Object.entries(recommendation.environment).map(([key, value]) => `  ${key}: "${value}"`).join('\n')}` : ''}\n`;
  const task = recommendation.task as Record<string, unknown>;
  const hardware = recommendation.hardware as Record<string, unknown>;
  return <div className="fixed inset-0 z-50 flex justify-end bg-black/20" onClick={onClose}>
    <aside role="dialog" aria-modal="true" aria-labelledby="recommended-settings" onClick={(event) => event.stopPropagation()} className="flex h-full w-full max-w-[640px] flex-col overflow-hidden border-l border-divider bg-panel shadow-2xl">
      <header className="flex items-start gap-3 border-b border-divider px-5 py-4">
        <div className="min-w-0 flex-1">
          <p className="type-eyebrow">SETTINGS CALCULATOR</p>
          <h2 id="recommended-settings" className="mt-1 font-serif text-2xl font-normal text-ink">Recommended settings</h2>
          <p className="mt-1 text-[11px] leading-5 text-muted">Sized from the checkpoint’s architecture, the target’s memory and this run’s token budget and concurrency. Values are estimates to verify at engine start.</p>
        </div>
        <button type="button" onClick={onClose} aria-label="Close recommended settings" className="rounded p-1 text-muted hover:bg-subtle hover:text-ink"><X size={18} /></button>
      </header>
      {recommendations.length > 1 && <div role="tablist" className="flex gap-1 border-b border-divider px-5 pt-2">{recommendations.map((item, index) => <button key={item.role} role="tab" aria-selected={index === active} onClick={() => setActive(index)} className={`border-b-2 px-2 pb-2 text-[11px] ${index === active ? 'border-violet-700 text-ink' : 'border-transparent text-muted hover:text-ink'}`}>{humanize(item.role)}</button>)}</div>}
      <div className="flex-1 space-y-5 overflow-y-auto px-5 py-4">
        {recommendation.state === 'unavailable'
          ? <p className="rounded border border-amber-200 bg-amber-50 px-3 py-2 text-[12px] text-amber-900">No recommendation for {humanize(recommendation.role)}: {recommendation.unavailable_reason}</p>
          : <>
            <dl className="grid grid-cols-2 gap-px overflow-hidden rounded border border-divider bg-divider text-[11px] sm:grid-cols-4">
              {[
                ['Binding', recommendation.binding_id ?? '—'],
                ['Hardware', `${hardware.accelerator ?? 'GPU'} · ${hardware.memory_gb ?? '?'} GB${Number(hardware.gpus) > 1 ? ` × ${hardware.gpus}` : ''}`],
                ['Load', `${task.purpose} · ${task.concurrency} in flight`],
                ['Tokens', `${Number(task.prompt_tokens).toLocaleString()} + ${Number(task.completion_tokens).toLocaleString()}`],
              ].map(([label, value]) => <div key={label} className="min-w-0 bg-surface px-3 py-2"><dt className="type-label">{label}</dt><dd title={String(value)} className="mt-1 truncate font-mono text-[10px] text-ink">{value}</dd></div>)}
            </dl>
            <section>
              <div className="mb-2 flex items-center gap-2">
                <h3 className="text-[12px] font-semibold text-ink">Engine</h3>
                <label className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-secondary"><input type="checkbox" checked={changesOnly} onChange={(event) => setChangesOnly(event.target.checked)} className="accent-violet-700" />Only differences</label>
                <button type="button" onClick={() => { void navigator.clipboard?.writeText(yaml); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }} className="inline-flex items-center gap-1 rounded border border-divider px-2 py-1 text-[11px] text-secondary hover:bg-subtle">{copied ? <Check size={12} /> : <Copy size={12} />}{copied ? 'Copied' : 'Copy YAML'}</button>
              </div>
              {rows.length === 0
                ? <p className="rounded bg-emerald-50 px-3 py-2 text-[11px] text-emerald-800">This run already uses every recommended engine value.</p>
                : <div className="overflow-hidden rounded border border-divider">
                  <table className="w-full text-left text-[11px]">
                    <thead className="bg-subtle text-[10px] uppercase tracking-wide text-muted"><tr><th className="px-3 py-1.5 font-medium">Setting</th><th className="px-3 py-1.5 font-medium">This run</th><th className="px-3 py-1.5 font-medium">Recommended</th></tr></thead>
                    <tbody className="divide-y divide-divider bg-surface">
                      {rows.map((row) => <tr key={row.key} className="align-top">
                        <td className="px-3 py-2"><code className="font-mono text-[11px] text-ink">{row.key}</code><p className="mt-1 text-[10px] leading-4 text-muted">{row.reason}</p></td>
                        <td className="px-3 py-2 font-mono text-[11px] text-secondary">{row.current === undefined || row.current === null ? <span className="italic text-muted">not set</span> : formatValue(row.current)}</td>
                        <td className={`px-3 py-2 font-mono text-[11px] ${row.changed ? 'font-semibold text-violet-700' : 'text-secondary'}`}>{formatValue(row.suggested)}</td>
                      </tr>)}
                    </tbody>
                  </table>
                </div>}
            </section>
            <section>
              <h3 className="mb-2 text-[12px] font-semibold text-ink">Memory per GPU</h3>
              <MemoryBar memory={recommendation.memory_gb} />
              <p className="mt-2 text-[11px] text-secondary">{recommendation.max_concurrency != null && <>{recommendation.max_concurrency.toLocaleString()} typical-length sequences fit in the KV budget. </>}{recommendation.decode_tokens_per_s_upper_bound != null && <>Decode upper bound ≈ {Math.round(recommendation.decode_tokens_per_s_upper_bound).toLocaleString()} tok/s (memory bandwidth).</>}</p>
            </section>
            {recommendation.step && <p className="text-[11px] leading-5 text-secondary">Step sizing: {recommendation.step.reason}. The options table on the page shows waves and estimated step time for each choice.</p>}
            {recommendation.notes.length > 0 && <section><h3 className="mb-1.5 text-[12px] font-semibold text-ink">Notes</h3><ul className="list-disc space-y-1 pl-4 text-[11px] leading-5 text-secondary">{recommendation.notes.map((note) => <li key={note}>{note}</li>)}</ul></section>}
          </>}
      </div>
    </aside>
  </div>;
}

export function ConfigPage({
  jobKind,
  jobDefinition,
  stage,
  schemaVersion,
  inputs,
  source,
  review,
  groups,
  labels,
  heading,
  sourceFields,
}: {
  jobKind: string;
  jobDefinition: string;
  stage: string;
  schemaVersion: number;
  inputs: Record<string, unknown>;
  source: Record<string, unknown>;
  review: ConfigurationReview | null | undefined;
  groups: ConfigGroup[];
  labels: Record<string, string>;
  heading: ReactNode;
  sourceFields: ReactNode;
}) {
  const [panelOpen, setPanelOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const [flaggedOnly, setFlaggedOnly] = useState(false);
  const findings = useMemo(() => review?.findings ?? [], [review]);
  const recommendations = useMemo(() => review?.recommendations ?? [], [review]);

  const suggestions = useMemo(() => {
    const map = new Map<string, unknown>();
    for (const recommendation of recommendations) {
      for (const setting of recommendation.settings ?? []) {
        if (setting.changed) map.set(`${recommendation.role}.engine.${setting.key}`, setting.suggested);
      }
    }
    return map;
  }, [recommendations]);

  const cards = useMemo(() => {
    const byRole = new Map<string, Card>();
    for (const [role, raw] of Object.entries(inputs)) {
      const card = cardFor(role, raw, labels);
      if (card) byRole.set(role, card);
    }
    // Suggestions for engine keys the run did not set, and findings on values the run did not record.
    for (const [path, suggested] of suggestions) {
      const role = path.split('.')[0];
      const card = byRole.get(role);
      if (!card) continue;
      for (const section of card.sections) for (const row of section.rows) if (row.path === path) row.suggested = suggested;
      if (!card.sections.some((section) => section.rows.some((row) => row.path === path))) {
        const sectionPath = path.split('.').slice(0, 2).join('.');
        const section = card.sections.find((item) => item.path === sectionPath);
        const row = { path, key: path.slice(sectionPath.length + 1), value: undefined, suggested };
        if (section) section.rows.push(row);
      }
    }
    for (const finding of findings) {
      const card = byRole.get(finding.role) ?? byRole.get(finding.path.split('.')[0]);
      if (!card) continue;
      const known = card.sections.some((section) => section.path === finding.path || section.rows.some((row) => row.path === finding.path));
      if (!known) {
        const general = card.sections.find((section) => section.path === card.role) ?? { path: card.role, title: 'Settings', rows: [] };
        if (!card.sections.includes(general)) card.sections.unshift(general);
        general.rows.push({ path: finding.path, key: finding.path.split('.').slice(1).join('.'), value: finding.value, missing: finding.value === null || finding.value === undefined });
      }
    }
    return byRole;
  }, [inputs, labels, suggestions, findings]);

  const findingsByPath = useMemo(() => {
    const map = new Map<string, ConfigurationFinding[]>();
    for (const finding of findings) {
      for (const path of new Set([finding.path, ...(finding.related_paths ?? [])])) map.set(path, [...(map.get(path) ?? []), finding]);
    }
    return map;
  }, [findings]);

  const anchoredPaths = useMemo(() => {
    const paths = new Set<string>();
    for (const card of cards.values()) {
      for (const section of card.sections) {
        paths.add(section.path);
        for (const row of section.rows) paths.add(row.path);
      }
    }
    return paths;
  }, [cards]);

  const assigned = new Set(groups.flatMap((group) => group.keys));
  const layout = [...groups, { title: 'Additional selections', description: 'Recorded selections outside this job kind’s curated layout.', keys: [...cards.keys()].filter((key) => !assigned.has(key)) }]
    .map((group) => ({ ...group, cards: group.keys.flatMap((key) => cards.get(key) ?? []) }))
    .filter((group) => group.cards.length > 0);
  const normalizedFilter = filter.trim().toLowerCase();
  const stepRecommendation = recommendations.find((item) => item.step);

  return <>
    <div className="flex flex-wrap items-start gap-4">
      <div className="min-w-0 flex-1">{heading}</div>
      {recommendations.length > 0 && <button type="button" onClick={() => setPanelOpen(true)} className="mt-6 inline-flex items-center gap-1.5 rounded-md border border-violet-200 bg-surface px-3 py-2 text-[12px] font-medium text-violet-700 shadow-sm hover:bg-violet-50"><SlidersHorizontal size={15} aria-hidden="true" />Recommended settings</button>}
    </div>
    <section aria-label="Run contract" className="mt-4 flex flex-wrap gap-x-5 gap-y-1 text-[11px] text-muted">
      {[['Job kind', jobKind], ['Definition', jobDefinition], ['Stage', stage], ['View schema', `v${schemaVersion}`]].map(([label, value]) => <span key={label}>{label} <code className="ml-1 font-mono text-[11px] text-ink">{value}</code></span>)}
    </section>
    {review && <ReviewSummary review={review} anchoredPaths={anchoredPaths} onRecommend={recommendations.length ? () => setPanelOpen(true) : null} />}
    {stepRecommendation?.step && <StepSizing step={stepRecommendation.step} calibration={review?.calibration} role={stepRecommendation.role} />}
    {cards.size > 0 && <div className="mt-8 grid gap-8 xl:grid-cols-[180px_minmax(0,1fr)]">
      <nav aria-label="Selections" className="hidden xl:block">
        <div className="sticky top-4 space-y-4">
          {layout.map((group) => <div key={group.title}>
            <p className="type-label mb-1.5">{group.title}</p>
            <ul className="space-y-0.5">{group.cards.map((card) => {
              const cardFindings = findings.filter((finding) => finding.path === card.role || finding.path.startsWith(`${card.role}.`));
              const severity = worst(cardFindings);
              return <li key={card.role}><a href={`#cfg-${card.role}`} onClick={(event) => { event.preventDefault(); scrollToPath(card.role); }} className="flex items-center gap-2 rounded px-2 py-1 text-[11px] text-secondary hover:bg-subtle hover:text-ink">{severity ? <span className={`h-1.5 w-1.5 shrink-0 rounded-full ${SEVERITY[severity].dot}`} /> : <span className="h-1.5 w-1.5 shrink-0" />}<span className="truncate">{card.label}</span>{cardFindings.length > 0 && <span className="ml-auto text-[10px] text-muted">{cardFindings.length}</span>}</a></li>;
            })}</ul>
          </div>)}
        </div>
      </nav>
      <div className="min-w-0">
        <div className="sticky top-0 z-10 -mx-1 mb-4 flex flex-wrap items-center gap-3 bg-canvas/95 px-1 py-2 backdrop-blur">
          <label className="relative flex min-w-[220px] flex-1 items-center">
            <MagnifyingGlass size={14} className="pointer-events-none absolute left-2.5 text-muted" aria-hidden="true" />
            <input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Filter settings by key or value" aria-label="Filter settings" className="w-full rounded-md border border-divider bg-surface py-1.5 pl-8 pr-3 text-[12px] text-ink placeholder:text-muted" />
          </label>
          <label className="inline-flex items-center gap-1.5 text-[11px] text-secondary"><input type="checkbox" checked={flaggedOnly} onChange={(event) => setFlaggedOnly(event.target.checked)} className="accent-violet-700" />Flagged and suggested only</label>
        </div>
        <div className="space-y-8">
          {layout.map((group) => <section key={group.title} aria-labelledby={`config-group-${group.title.replaceAll(' ', '-').toLowerCase()}`}>
            <h2 id={`config-group-${group.title.replaceAll(' ', '-').toLowerCase()}`} className="font-serif text-xl font-normal text-ink">{group.title}</h2>
            <p className="mb-3 mt-0.5 text-xs leading-5 text-muted">{group.description}</p>
            <div className="space-y-3">{group.cards.map((card) => <ConfigCard key={card.role} card={card} findingsByPath={findingsByPath} filter={normalizedFilter} flaggedOnly={flaggedOnly} />)}</div>
          </section>)}
        </div>
      </div>
    </div>}
    {Object.keys(source).length > 0 && <section aria-labelledby="source-provenance" className="mt-10">
      <h2 id="source-provenance" className="font-serif text-xl font-normal text-ink">Source provenance</h2>
      <p className="mb-3 mt-0.5 text-xs leading-5 text-muted">The code revision and working-tree state recorded when the run started.</p>
      {sourceFields}
    </section>}
    <details className="obs-card mt-8 overflow-hidden">
      <summary className="cursor-pointer select-none px-4 py-3 text-[11px] font-medium text-secondary hover:bg-subtle">View redacted JSON</summary>
      <pre className="max-h-[520px] overflow-auto border-t border-divider bg-subtle p-4 font-mono text-[10px] leading-5 text-secondary">{JSON.stringify({ selections: inputs, source_metadata: source }, null, 2)}</pre>
    </details>
    {panelOpen && recommendations.length > 0 && <RecommendationPanel recommendations={recommendations} onClose={() => setPanelOpen(false)} />}
  </>;
}

export function ConfigurationStrip({ review, onOpen }: { review: ConfigurationReview | null | undefined; onOpen: () => void }) {
  const findings = review?.findings ?? [];
  const counts = severityCounts(findings);
  const active = SEVERITY_ORDER.filter((severity) => severity !== 'info' && counts[severity] > 0);
  if (!review || active.length === 0) return null;
  const top = active[0];
  return <section aria-label="Configuration review summary" className={`obs-card mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 border-l-2 px-4 py-2.5 text-[11px] ${SEVERITY[top].edge}`}>
    <span className={`inline-flex items-center gap-1.5 font-medium ${SEVERITY[top].text}`}>{SEVERITY[top].icon}Configuration</span>
    {active.map((severity) => <span key={severity} className="inline-flex items-center gap-1.5 text-secondary"><span className={`h-1.5 w-1.5 rounded-full ${SEVERITY[severity].dot}`} />{counts[severity]} {counts[severity] === 1 ? SEVERITY[severity].label.toLowerCase() : SEVERITY[severity].plural}</span>)}
    <span className="min-w-0 flex-1 truncate text-muted" title={findings.find((finding) => finding.severity === top)?.message}>{findings.find((finding) => finding.severity === top)?.code}</span>
    <button type="button" onClick={onOpen} className="inline-flex items-center gap-1 text-violet-700 hover:underline">Review configuration<ArrowRight size={11} aria-hidden="true" /></button>
  </section>;
}
