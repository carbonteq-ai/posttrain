import { useEffect, useMemo, useRef, useState } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import type { TraceEvaluation } from '../lib/api';
import type { TracePresentation } from '../lib/trace-presentation';

echarts.use([BarChart, ScatterChart, GridComponent, TooltipComponent, CanvasRenderer]);

function MiniChart({ option, label }: { option: echarts.EChartsCoreOption; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const chart = echarts.init(ref.current, undefined, { renderer: 'canvas' });
    chart.setOption({ animation: false, ...option });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(ref.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [option]);
  return <div ref={ref} className="h-[190px] w-full" role="img" aria-label={label} />;
}

const axis = {
  axisLine: { lineStyle: { color: '#aaa4ab' } },
  axisTick: { show: false },
  axisLabel: { color: '#777179', fontSize: 11 },
  splitLine: { lineStyle: { color: '#ebe7e1', type: 'dashed' } },
};

export function EvaluationCharts({ evaluation, presentation }: { evaluation: TraceEvaluation; presentation: TracePresentation }) {
  const [scatterMetric, setScatterMetric] = useState<'tool_calls' | 'response_chars' | 'thinking_chars'>('tool_calls');
  const [breakdownMetric, setBreakdownMetric] = useState<'success_rate' | 'mean_reward'>(
    presentation.defaultBreakdownMetric,
  );
  useEffect(() => {
    setBreakdownMetric(presentation.defaultBreakdownMetric);
  }, [presentation.defaultBreakdownMetric]);
  const passRateAvailable = presentation.passRateAvailable;
  const facetDimensions = [...new Set(evaluation.facets.map((facet) => facet.dimension))];
  const multipleFacetDimensions = facetDimensions.length > 1;
  const facetLabel = (facet: TraceEvaluation['facets'][number]) => multipleFacetDimensions
    ? `${facet.dimension_label} · ${facet.label}`
    : facet.label;
  const breakdownTitle = evaluation.facets.length
    ? `${multipleFacetDimensions ? 'capability facet' : evaluation.facets[0].dimension_label.toLowerCase()}`
    : 'task slice';
  const options = useMemo(() => {
    const breakdown = evaluation.facets.length ? evaluation.facets : evaluation.slices;
    const rewards = evaluation.traces.flatMap((trace) => (trace.reward == null ? [] : [trace.reward]));
    const minimum = rewards.length ? Math.floor(Math.min(...rewards) * 5) / 5 : 0;
    const maximum = rewards.length ? Math.ceil(Math.max(...rewards) * 5) / 5 : 1;
    const binCount = 12;
    const bins = Array.from({ length: binCount }, (_, index) => {
      const start = minimum + ((maximum - minimum || 1) / binCount) * index;
      const end = minimum + ((maximum - minimum || 1) / binCount) * (index + 1);
      return {
        label: start.toFixed(1),
        count: rewards.filter((reward) => reward >= start && (index === binCount - 1 ? reward <= end : reward < end)).length,
      };
    });
    const breakdownValues = breakdown.map((slice) => breakdownMetric === 'success_rate' ? slice.success_rate : slice.mean_reward)
      .filter((value): value is number => value != null);
    const breakdownMinimum = breakdownMetric === 'success_rate'
      ? 0
      : breakdownValues.length ? Math.floor(Math.min(...breakdownValues) * 10) / 10 : 0;
    const breakdownMaximum = breakdownMetric === 'success_rate'
      ? 100
      : breakdownValues.length ? Math.ceil(Math.max(...breakdownValues) * 10) / 10 || 1 : 1;
    const breakdownFormatter = breakdownMetric === 'success_rate'
      ? (value: number) => `${value}%`
      : (value: number) => value.toFixed(2);
    return {
      rewards: {
        grid: { left: 35, right: 12, top: 12, bottom: 28 },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: bins.map((bin) => bin.label), ...axis },
        yAxis: { type: 'value', ...axis },
        series: [{ type: 'bar', data: bins.map((bin) => bin.count), itemStyle: { color: '#5bbcb7' }, barMaxWidth: 16, barCategoryGap: '58%' }],
      },
      slices: {
        grid: { left: 82, right: 28, top: 12, bottom: 28 },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value', min: breakdownMinimum, max: breakdownMaximum, ...axis, axisLabel: { ...axis.axisLabel, formatter: breakdownFormatter } },
        yAxis: { type: 'category', data: breakdown.map((slice) => 'dimension_label' in slice ? facetLabel(slice) : slice.label), ...axis },
        series: [{
          type: 'bar',
          data: breakdown.map((slice) => {
            const value = breakdownMetric === 'success_rate' ? slice.success_rate : slice.mean_reward;
            return value == null ? null : breakdownMetric === 'success_rate' ? Math.round(value * 1000) / 10 : Math.round(value * 1000) / 1000;
          }),
          itemStyle: { color: '#5bbcb7' },
          barMaxWidth: 18,
          label: { show: true, position: 'right', formatter: (params: { value: number | null }) => params.value == null ? '—' : breakdownMetric === 'success_rate' ? `${params.value}%` : Number(params.value).toFixed(2), color: '#5f5961', fontSize: 11 },
        }],
      },
      scatter: {
        grid: { left: 40, right: 16, top: 12, bottom: 28 },
        tooltip: { trigger: 'item' },
        xAxis: {
          type: 'value',
          name: scatterMetric === 'tool_calls' ? 'Tool calls' : scatterMetric === 'response_chars' ? 'Response characters' : 'Thinking characters',
          nameLocation: 'middle',
          nameGap: 22,
          ...axis,
        },
        yAxis: { type: 'value', name: 'Reward', ...axis },
        series: (['pass', 'review', 'scored', 'error', 'truncated', 'unknown'] as const).map((outcome) => {
          const groups = new Map<string, { x: number; y: number; count: number }>();
          evaluation.traces
            .filter((trace) => trace.outcome === outcome && trace.reward != null && trace[scatterMetric] != null)
            .forEach((trace) => {
              const x = trace[scatterMetric] as number;
              const y = trace.reward as number;
              const key = `${x}:${y}`;
              const current = groups.get(key);
              groups.set(key, current ? { ...current, count: current.count + 1 } : { x, y, count: 1 });
            });
          return {
          name: outcome === 'review' ? 'Needs review' : outcome[0].toUpperCase() + outcome.slice(1),
          type: 'scatter' as const,
          symbolSize: 7,
          itemStyle: { color: { pass: '#12a5a1', review: '#e55f4a', scored: '#7a66d9', error: '#b42318', truncated: '#d28a18', unknown: '#8a858b' }[outcome] },
          data: Array.from(groups.values()).map((group) => ({
            value: [group.x, group.y],
            symbolSize: group.count > 1 ? Math.min(24, 7 + Math.sqrt(group.count) * 4) : 7,
            label: group.count > 1 ? { show: true, formatter: `×${group.count}`, color: '#5f5961', fontSize: 10 } : undefined,
          })),
        };
        }),
      },
    };
  }, [evaluation, facetLabel, scatterMetric]);

  const plotted = evaluation.traces.filter((trace) => trace.reward != null && trace[scatterMetric] != null).length;
  const missing = evaluation.traces.filter((trace) => trace.reward == null || trace[scatterMetric] == null).length;
  const scatterLabel = scatterMetric === 'tool_calls' ? 'Tool calls' : scatterMetric === 'response_chars' ? 'Response length' : 'Thinking output';

  return (
    <div className="obs-card grid overflow-hidden divide-x divide-divider lg:grid-cols-3">
      <section className="p-4">
        <h3 className="text-[13px] font-medium">Reward distribution</h3>
        <p className="mt-1 text-xs text-muted">{evaluation.included} trace records</p>
        <MiniChart option={options.rewards} label="Histogram of trace rewards" />
      </section>
      <section className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h3 className="text-[13px] font-medium">{breakdownMetric === 'success_rate' ? 'Pass rate' : 'Mean reward'} by {breakdownTitle}</h3>
            <p className="mt-1 text-xs text-muted">{breakdownMetric === 'success_rate'
              ? evaluation.metadata?.pass_rate_basis ?? 'Configured verifier success signal'
              : evaluation.facets.length ? 'Mean native reward across each semantic facet' : 'Mean native reward across each task slice'}</p>
          </div>
          <div className="flex rounded-[4px] border border-divider p-0.5 text-[10px]" role="group" aria-label="Slice breakdown metric">
            <button type="button" aria-pressed={breakdownMetric === 'success_rate'} disabled={!passRateAvailable} onClick={() => setBreakdownMetric('success_rate')} className={`rounded-[3px] px-2 py-1 ${breakdownMetric === 'success_rate' ? 'bg-violet-700 text-white' : 'text-muted hover:text-ink disabled:cursor-not-allowed disabled:opacity-50'}`}>Pass rate</button>
            <button type="button" aria-pressed={breakdownMetric === 'mean_reward'} onClick={() => setBreakdownMetric('mean_reward')} className={`rounded-[3px] px-2 py-1 ${breakdownMetric === 'mean_reward' ? 'bg-violet-700 text-white' : 'text-muted hover:text-ink'}`}>Mean reward</button>
          </div>
        </div>
        <MiniChart option={options.slices} label={`${breakdownMetric === 'success_rate' ? 'Pass rate' : 'Mean reward'} by evaluation slice`} />
      </section>
      <section className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h3 className="text-[13px] font-medium">Reward vs. {scatterLabel.toLowerCase()}</h3>
            <p className="mt-1 text-xs text-muted">{plotted} plotted{missing ? ` · ${missing} missing` : ''}</p>
          </div>
          <div className="flex rounded-[4px] border border-divider p-0.5 text-[10px]" role="group" aria-label="Scatter plot x-axis">
            {([
              ['tool_calls', 'Tools'],
              ['response_chars', 'Response'],
              ['thinking_chars', 'Thinking'],
            ] as const).map(([value, label]) => <button key={value} type="button" onClick={() => setScatterMetric(value)} className={`rounded-[3px] px-2 py-1 ${scatterMetric === value ? 'bg-violet-700 text-white' : 'text-muted hover:text-ink'}`}>{label}</button>)}
          </div>
        </div>
        <MiniChart option={options.scatter} label={`Trace rewards plotted against ${scatterLabel.toLowerCase()}`} />
      </section>
    </div>
  );
}
