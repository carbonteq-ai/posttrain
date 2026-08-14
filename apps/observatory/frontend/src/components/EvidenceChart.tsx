import { useEffect, useMemo, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import type { MetricSeries } from '../lib/api';

echarts.use([
  BarChart,
  LineChart,
  ScatterChart,
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TitleComponent,
  TooltipComponent,
  CanvasRenderer,
]);

const colors = ['#6356c7', '#e85d3f', '#148c87', '#c68a2c', '#2476a8', '#8f548f'];
const lineTypes = ['solid', 'dashed', 'dotted'] as const;
export type ChartXDomain = 'logical-step' | 'elapsed-time';

function shortName(name: string, metricLabels: Record<string, string>): string {
  return metricLabels[name] ?? name.split('/').at(-1)?.replaceAll('_', ' ') ?? name;
}

type AxisTooltipParam = {
  axisValue?: number | string;
  color?: string;
  data?: unknown;
  seriesName?: string;
  value?: unknown;
};

type AxisPointerEvent = {
  axesInfo?: Array<{ value?: unknown }>;
};

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (character) => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  })[character] ?? character);
}

function tooltipNumber(param: AxisTooltipParam): number | null {
  const value = Array.isArray(param.value) ? param.value[1] : param.value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  const data = Array.isArray(param.data) ? param.data[1] : param.data;
  return typeof data === 'number' && Number.isFinite(data) ? data : null;
}

function tooltipObservedAt(param: AxisTooltipParam): number | null {
  if (!Array.isArray(param.data)) return null;
  const value = param.data[2];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

export function formatElapsedDuration(value: number): string {
  const seconds = Math.max(0, Math.round(value));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m${remainingSeconds ? ` ${remainingSeconds}s` : ''}`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  if (hours < 24) return `${hours}h${remainingMinutes ? ` ${remainingMinutes}m` : ''}`;
  const days = Math.floor(hours / 24);
  const remainingHours = hours % 24;
  return `${days}d${remainingHours ? ` ${remainingHours}h` : ''}`;
}

export function formatElapsedAxis(value: number, maximum: number): string {
  if (maximum < 120) return `${Math.round(value)}s`;
  if (maximum < 60 * 60) return `${Math.round(value / 60)}m`;
  if (maximum < 2 * 24 * 60 * 60) return `${Number((value / 3600).toFixed(1))}h`;
  return `${Number((value / 86400).toFixed(1))}d`;
}

function formatTooltipValue(value: number | null, unit?: string | null): string {
  if (value == null) return '—';
  if (unit === 'bytes') {
    const gib = value / 1024 ** 3;
    return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
  }
  if (unit === 'ratio') return `${(value * 100).toFixed(1)}%`;
  if (unit === '%') return `${value.toFixed(0)}%`;
  if (unit === 's') {
    return value >= 60
      ? `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`
      : `${value.toFixed(value >= 10 ? 1 : 2)}s`;
  }
  if (unit === 'GiB') return `${value.toFixed(value >= 10 ? 1 : 2)} GiB`;
  if (unit === 'samples/s') return `${value.toFixed(2)} samples/s`;
  if (unit === 'tokens/s') return `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })} tokens/s`;
  if (unit === 'tokens') return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tokens`;
  if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function formatTooltip(
  params: AxisTooltipParam | AxisTooltipParam[],
  metricLabels: Record<string, string>,
  metricUnits: Record<string, string | null>,
  xDomain: ChartXDomain = 'logical-step',
): string {
  const entries = (Array.isArray(params) ? params : [params]).filter((param) => param.seriesName);
  if (entries.length === 0) return '';
  const axisValue = entries[0].axisValue;
  const step = typeof axisValue === 'number' ? axisValue : Number(axisValue);
  const observedAt = tooltipObservedAt(entries[0]);
  const header = xDomain === 'elapsed-time'
    ? `${observedAt == null ? 'Time unavailable' : new Date(observedAt).toLocaleString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
      })}${Number.isFinite(step) ? ` · +${formatElapsedDuration(step)}` : ''}`
    : `Step ${Number.isFinite(step) ? step.toLocaleString(undefined, { maximumFractionDigits: 0 }) : String(axisValue ?? '—')}`;
  const rows = entries.map((param) => {
    const name = param.seriesName ?? '';
    const color = typeof param.color === 'string' ? param.color : '#716a73';
    const label = shortName(name, metricLabels);
    const value = formatTooltipValue(tooltipNumber(param), metricUnits[name]);
    return `<div style="display:grid;grid-template-columns:12px minmax(92px,1fr) auto;align-items:center;column-gap:7px;padding-top:5px;">`
      + `<span aria-hidden="true" style="display:block;width:11px;height:2px;border-radius:2px;background:${escapeHtml(color)};"></span>`
      + `<span style="min-width:0;color:#5f5a62;font-size:10px;font-weight:400;line-height:1.35;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escapeHtml(label)}</span>`
      + `<span style="padding-left:8px;color:#262126;font-size:10px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.35;text-align:right;white-space:nowrap;">${escapeHtml(value)}</span>`
      + '</div>';
  }).join('');
  return `<div style="min-width:190px;max-width:320px;padding:9px 10px 10px;font-family:var(--font-sans);">`
    + `<div style="padding-bottom:2px;color:#817a83;font-size:9px;font-weight:500;letter-spacing:.04em;line-height:1.2;">${escapeHtml(header)}</div>`
    + rows
    + '</div>';
}

/**
 * Return the scale family a metric can honestly share. A chart definition says
 * the series belong to one analytical question; this function only decides
 * whether their numeric axes are compatible.
 */
export function scaleGroup(name: string, metricUnits: Record<string, string | null>): string {
  if (name === 'train/loss' || name === 'train/validation/loss') return 'loss';
  if (name.startsWith('train/rl/reward_')) return 'rl-reward';
  if (name === 'train/mean_token_accuracy' || name === 'train/gradient_clipped' || name === 'train/rewards/accuracies') return 'ratio';
  if (name.startsWith('train/rewards/') && name !== 'train/rewards/accuracies') return 'dpo-reward';
  if (name.startsWith('train/logps/')) return 'dpo-log-probability';
  if (name.startsWith('train/logits/')) return 'dpo-logit';
  if (/train\/rl\/rollouts_(requested|attempted|completed|failed|truncated|unscorable|missing)$/.test(name)) return 'rollout-count';
  if (/train\/rl\/active_sampling_(generation_rounds|generated_rows|candidate_groups_(reserved|generated|retained|unused))$/.test(name)) return 'active-sampling-count';
  if (name === 'trace/rollout/avg_thinking_tokens' || name === 'trace/rollout/avg_output_tokens') return 'rollout-tokens';
  if (name === 'trace/rollout/avg_tool_calls') return 'rollout-tool-calls';
  if (name === 'train/learning_rate') return 'learning-rate';
  if (name === 'train/non_padding_tokens_per_second') return 'tokens-per-second';
  if (name === 'train/step_time_seconds') return 'seconds';
  const unit = metricUnits[name];
  if (unit) return `unit:${unit}`;
  if (/(fraction|_ratio|_rate)$/.test(name)) return 'ratio';
  if (/(used_bytes|rss_bytes|memory_bytes)$/.test(name)) return 'unit:bytes';
  if (/(gpu_utilization|cpu_percent)$/.test(name)) return 'unit:%';
  return name;
}

/**
 * Keep conceptually coupled policy-update signals in one divided panel even
 * when they need independent numeric axes. This controls chart composition,
 * not whether two values share a ruler, and applies to every job family that
 * records the standard GRPO-style policy loss and entropy metrics.
 */
function panelGroup(name: string, metricUnits: Record<string, string | null>): string {
  if (name === 'train/rl/policy_loss' || name === 'train/rl/entropy') return 'policy-update';
  if (name.startsWith('trace/rollout/avg_')) return 'rollout-behavior';
  return scaleGroup(name, metricUnits);
}

function axisFormatter(group: string) {
  if (group === 'unit:bytes' || group.includes('bytes')) {
    return (value: number) => `${(value / 1024 ** 3).toFixed(1)} GiB`;
  }
  if (group === 'ratio' || group === 'unit:ratio') {
    return (value: number) => `${(value * 100).toFixed(0)}%`;
  }
  if (group === 'unit:%') return (value: number) => `${value.toFixed(0)}%`;
  return undefined;
}

function axisBounds(group: string) {
  if (group === 'ratio' || group === 'unit:ratio') {
    return {
      min: ({ min }: { min: number }) => min < 0 ? Math.floor(min * 20) / 20 : 0,
      max: ({ max }: { max: number }) => max > 1
        ? Math.ceil(max * 20) / 20
        : Math.max(0.05, Math.ceil((max + 0.025) * 20) / 20),
    };
  }
  if (group === 'unit:%') {
    return {
      min: ({ min }: { min: number }) => min < 0 ? Math.floor(min / 10) * 10 : 0,
      max: ({ max }: { max: number }) => max > 100 ? Math.ceil(max / 10) * 10 : 100,
    };
  }
  if (group === 'rollout-count') return { min: 0 };
  return { scale: true };
}

function groupLabel(
  group: string,
  items: MetricSeries[],
  metricLabels: Record<string, string>,
): string {
  if (items.length === 1) return shortName(items[0].name, metricLabels);
  return {
    loss: 'Training and validation loss',
    'rl-reward': 'Reward signal',
    'dpo-reward': 'Preference rewards',
    'dpo-log-probability': 'Sequence log probability',
    'dpo-logit': 'Mean token logits',
    'rollout-count': 'Rollout count',
    'rollout-behavior': 'Rollout behavior',
    'policy-update': 'Policy update and exploration',
    'active-sampling-count': 'Candidate rows',
    ratio: 'Rates and fractions',
    'unit:ratio': 'Rates and fractions',
    'unit:%': 'Utilization',
    'unit:bytes': 'Memory',
  }[group] ?? group.replace(/^unit:/, '').replaceAll('-', ' ');
}

type EvidenceChartProps = {
  series: MetricSeries[];
  height?: number;
  compact?: boolean;
  ariaLabel: string;
  metricLabels?: Record<string, string>;
  metricUnits?: Record<string, string | null>;
  showLegend?: boolean;
  selectedStep?: number | null;
  onPointSelect?: (step: number) => void;
  hoveredStep?: number | null;
  onHoverStep?: (step: number | null) => void;
  xDomain?: ChartXDomain;
  xOrigin?: string;
  xRange?: readonly [number, number];
};

export function EvidenceChart({
  series,
  height = 330,
  compact = false,
  ariaLabel,
  metricLabels = {},
  metricUnits = {},
  showLegend = true,
  selectedStep,
  onPointSelect,
  hoveredStep,
  onHoverStep,
  xDomain = 'logical-step',
  xOrigin,
  xRange,
}: EvidenceChartProps) {
  const elementRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const applyingSharedHoverRef = useRef(false);
  const pointerInsideRef = useRef(false);
  const configuredXOriginMs = useMemo(() => xOrigin == null ? null : Date.parse(xOrigin), [xOrigin]);
  const plottedSeries = useMemo(
    () => series
      .filter((item) => item.points.length > 0)
      .map((item) => ({
        ...item,
        // Observatory's service orders logical series. Retain this defensive
        // boundary so a future data source cannot make the canvas connect
        // points backward while the surrounding summaries remain correct.
        points: [...item.points]
          .filter((point) => xDomain === 'logical-step' || (point.observed_at != null && Number.isFinite(Date.parse(point.observed_at))))
          .sort((left, right) => xDomain === 'elapsed-time'
            ? Date.parse(left.observed_at ?? '') - Date.parse(right.observed_at ?? '')
            : (left.step ?? -1) - (right.step ?? -1)),
      })),
    [series, xDomain],
  );
  const observedXOriginMs = Math.min(
    ...plottedSeries.flatMap((item) => item.points.map((point) => Date.parse(point.observed_at ?? ''))),
  );
  const xOriginMs = configuredXOriginMs != null && Number.isFinite(configuredXOriginMs)
    ? configuredXOriginMs
    : Number.isFinite(observedXOriginMs) ? observedXOriginMs : null;
  const xMinimum = xDomain === 'elapsed-time' && xRange != null && xRange[1] > xRange[0]
    ? xRange[0]
    : undefined;
  const xMaximum = xDomain === 'elapsed-time' && xRange != null && xRange[1] > xRange[0]
    ? xRange[1]
    : undefined;
  const scaleGroups = useMemo(
    () => [...new Set(plottedSeries.map((item) => scaleGroup(item.name, metricUnits)))],
    [metricUnits, plottedSeries],
  );
  const panelGroups = useMemo(
    () => [...new Set(plottedSeries.map((item) => panelGroup(item.name, metricUnits)))],
    [metricUnits, plottedSeries],
  );
  // One or two scales remain overlaid so related evidence can be compared.
  // Three or more scales become synchronized small multiples instead of a
  // forest of rulers around one plot. A semantic panel can contain multiple
  // axes, such as policy loss and entropy, without splitting their evidence.
  const useSmallMultiples = scaleGroups.length > 2;
  const pointCount = Math.max(0, ...plottedSeries.map((item) => item.points.length));
  const elapsedMaximum = xDomain === 'elapsed-time' && xOriginMs != null && Number.isFinite(xOriginMs)
    ? Math.max(0, ...plottedSeries.flatMap((item) => item.points.map((point) => (Date.parse(point.observed_at ?? '') - xOriginMs) / 1000)))
    : 0;
  const showZoom = !compact && pointCount > 24;
  const renderedHeight = useSmallMultiples
    ? Math.max(height, 72 + panelGroups.length * 92 + (showZoom ? 34 : 0))
    : height;

  useEffect(() => {
    if (!elementRef.current) return;
    const chart = echarts.init(elementRef.current, undefined, { renderer: 'canvas' });
    chartRef.current = chart;
    const groupSeries = panelGroups.map((group) => plottedSeries.filter((item) => panelGroup(item.name, metricUnits) === group));
    const groupAxisGroups = groupSeries.map((items) => [...new Set(items.map((item) => scaleGroup(item.name, metricUnits)))]);
    const showChartLegend = showLegend && plottedSeries.length > 1;
    // The legend and the first panel title occupy the same upper chart area.
    // Reserve a small gutter so multi-series legends never read as the first
    // panel's data or label.
    const legendHeight = showChartLegend ? 46 : 8;
    const bottomSpace = compact ? (xDomain === 'elapsed-time' ? 44 : 34) : showZoom ? 66 : 44;
    const panelGap = 30;
    const panelCount = useSmallMultiples ? panelGroups.length : 1;
    const availablePanelHeight = renderedHeight - legendHeight - bottomSpace - panelGap * (panelCount - 1);
    const panelHeight = Math.max(54, Math.floor(availablePanelHeight / panelCount));
    const panelTops = Array.from(
      { length: panelCount },
      (_, index) => legendHeight + index * (panelHeight + panelGap),
    );
    const axes = useSmallMultiples
      ? panelGroups.flatMap((_, panelIndex) => groupAxisGroups[panelIndex].map((group, axisIndex) => ({
          type: 'value' as const,
          gridIndex: panelIndex,
          position: axisIndex === 0 ? ('left' as const) : ('right' as const),
          ...axisBounds(group),
          axisLine: axisIndex === 0
            ? { show: false }
            : { show: true, lineStyle: { color: colors[plottedSeries.findIndex((item) => scaleGroup(item.name, metricUnits) === group)] } },
          axisLabel: { color: '#817a83', fontSize: 10, formatter: axisFormatter(group) },
          splitLine: { show: axisIndex === 0, lineStyle: { color: '#e8e4de', type: 'dashed' as const } },
        })))
      : scaleGroups.map((group, index) => ({
          type: 'value' as const,
          position: index === 0 ? ('left' as const) : ('right' as const),
          ...axisBounds(group),
          axisLine: { show: index > 0, lineStyle: { color: colors[plottedSeries.findIndex((item) => scaleGroup(item.name, metricUnits) === group)] } },
          axisLabel: { color: '#817a83', fontSize: 10, formatter: axisFormatter(group) },
          splitLine: { show: index === 0, lineStyle: { color: '#e8e4de', type: 'dashed' as const } },
        }));
    const xAxes = useSmallMultiples
      ? panelGroups.map((_, index) => ({
          type: 'value' as const,
          scale: xDomain === 'elapsed-time',
          min: xMinimum,
          max: xMaximum,
          gridIndex: index,
          name: index === panelGroups.length - 1 && (!compact || xDomain === 'elapsed-time')
            ? (xDomain === 'elapsed-time' ? 'Elapsed run time' : 'Logical step')
            : '',
          nameLocation: 'middle' as const,
          nameGap: 32,
          axisLine: { lineStyle: { color: '#aaa4ab' } },
          axisTick: { show: false },
          axisLabel: {
            show: index === panelGroups.length - 1,
            color: '#78727a',
            fontSize: 11,
            formatter: xDomain === 'elapsed-time' ? (value: number) => formatElapsedAxis(value, elapsedMaximum) : undefined,
          },
          splitLine: { show: false },
        }))
      : [{
          type: 'value' as const,
          scale: xDomain === 'elapsed-time',
          min: xMinimum,
          max: xMaximum,
          name: compact && xDomain === 'logical-step' ? '' : (xDomain === 'elapsed-time' ? 'Elapsed run time' : 'Logical step'),
          nameLocation: 'middle' as const,
          nameGap: 34,
          axisLine: { lineStyle: { color: '#aaa4ab' } },
          axisTick: { show: false },
          axisLabel: {
            color: '#78727a',
            fontSize: 11,
            formatter: xDomain === 'elapsed-time' ? (value: number) => formatElapsedAxis(value, elapsedMaximum) : undefined,
          },
          splitLine: { show: false },
        }];
    const xAxisIndices = xAxes.map((_, index) => index);

    chart.setOption({
      animation: false,
      color: colors,
      title: useSmallMultiples
        ? panelGroups.map((group, index) => ({
            text: groupLabel(group, groupSeries[index], metricLabels),
            top: panelTops[index] - 20,
            left: compact ? 46 : 56,
            textStyle: { color: '#5f5a62', fontSize: 10, fontWeight: 500 },
          }))
        : [],
      grid: useSmallMultiples
        ? panelTops.map((top) => ({ left: compact ? 46 : 56, right: 24, top, height: panelHeight }))
        : {
            left: compact ? 46 : 56,
            right: axes.length > 1 ? 54 : 24,
            top: compact ? 32 : 44,
            bottom: bottomSpace,
          },
      tooltip: {
        trigger: 'axis',
        confine: true,
        renderMode: 'html',
        padding: 0,
        backgroundColor: '#fffdfa',
        borderColor: '#d8d3cb',
        borderWidth: 1,
        extraCssText: 'border-radius:5px;box-shadow:0 6px 18px rgba(38,33,38,.10);',
        axisPointer: { type: 'line', lineStyle: { color: '#8c858e', width: 1 } },
        formatter: (params: AxisTooltipParam | AxisTooltipParam[]) => formatTooltip(params, metricLabels, metricUnits, xDomain),
      },
      axisPointer: useSmallMultiples ? { link: [{ xAxisIndex: 'all' }] } : undefined,
      legend: {
        show: showChartLegend,
        type: 'scroll',
        top: 4,
        left: 0,
        itemWidth: 16,
        itemHeight: 2,
        itemGap: 18,
        textStyle: { color: '#5f5a62', fontSize: 10 },
        formatter: (name: string) => shortName(name, metricLabels),
      },
      xAxis: xAxes,
      yAxis: axes.length ? axes : [{ type: 'value' }],
      dataZoom: compact || !showZoom
        ? []
        : [
            { type: 'inside', xAxisIndex: xAxisIndices },
            {
              type: 'slider',
              xAxisIndex: xAxisIndices,
              height: 18,
              bottom: 10,
              borderColor: '#d8d3cb',
              backgroundColor: '#f7f5f1',
              fillerColor: 'rgba(99, 86, 199, .10)',
              handleStyle: { color: '#ffffff', borderColor: '#6356c7' },
              textStyle: { color: '#78727a', fontSize: 9 },
            },
          ],
      series: plottedSeries.map((item, seriesIndex) => {
        const scaleGroupIndex = Math.max(0, scaleGroups.indexOf(scaleGroup(item.name, metricUnits)));
        const panelGroupIndex = Math.max(0, panelGroups.indexOf(panelGroup(item.name, metricUnits)));
        const axisIndexWithinPanel = groupAxisGroups[panelGroupIndex]?.indexOf(scaleGroup(item.name, metricUnits)) ?? 0;
        const yAxisIndex = useSmallMultiples
          ? groupAxisGroups.slice(0, panelGroupIndex).reduce((count, groups) => count + groups.length, 0) + axisIndexWithinPanel
          : scaleGroupIndex;
        const indexWithinGroup = groupSeries[panelGroupIndex]?.findIndex((candidate) => candidate.name === item.name) ?? 0;
        return {
          name: item.name,
          type: 'line',
          xAxisIndex: useSmallMultiples ? panelGroupIndex : 0,
          yAxisIndex,
          showSymbol: item.points.length < 12,
          symbolSize: 5,
          smooth: false,
          lineStyle: { width: 1.9, type: lineTypes[indexWithinGroup % lineTypes.length] },
          emphasis: { focus: 'series' },
          markLine: selectedStep == null ? undefined : {
            silent: true,
            symbol: 'none',
            label: { show: false },
            lineStyle: { color: '#716a73', type: 'dashed', width: 1 },
            data: [{ xAxis: selectedStep }],
          },
          itemStyle: { color: colors[seriesIndex % colors.length] },
          data: item.points.map((point, pointIndex) => xDomain === 'elapsed-time'
            ? [
                xOriginMs == null || !Number.isFinite(xOriginMs)
                  ? pointIndex
                  : (Date.parse(point.observed_at ?? '') - xOriginMs) / 1000,
                point.value,
                Date.parse(point.observed_at ?? ''),
              ]
            : [point.step ?? pointIndex, point.value]),
        };
      }),
    });
    if (onPointSelect) {
      chart.on('click', (event) => {
        const data = event.data;
        if (Array.isArray(data) && typeof data[0] === 'number') onPointSelect(data[0]);
      });
    }
    if (onHoverStep) {
      chart.on('updateAxisPointer', (event) => {
        if (applyingSharedHoverRef.current || !pointerInsideRef.current) return;
        const pointerEvent = event as AxisPointerEvent;
        const value = pointerEvent.axesInfo?.find((axis) => typeof axis.value === 'number')?.value;
        if (typeof value === 'number' && Number.isFinite(value)) onHoverStep(value);
      });
      chart.getZr().on('mousemove', () => {
        pointerInsideRef.current = true;
      });
      chart.getZr().on('globalout', () => {
        pointerInsideRef.current = false;
        onHoverStep(null);
      });
    }
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(elementRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
      if (chartRef.current === chart) chartRef.current = null;
    };
  }, [compact, elapsedMaximum, metricLabels, metricUnits, onHoverStep, onPointSelect, panelGroups, plottedSeries, renderedHeight, scaleGroups, selectedStep, showLegend, showZoom, useSmallMultiples, xDomain, xMaximum, xMinimum, xOriginMs]);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    applyingSharedHoverRef.current = true;
    try {
      if (hoveredStep == null) {
        chart.dispatchAction({ type: 'updateAxisPointer', currTrigger: 'leave' });
        chart.dispatchAction({ type: 'hideTip' });
        return;
      }
      const anchor = plottedSeries.find((item) => item.points.length > 0);
      if (!anchor) return;
      let nearestIndex = 0;
      let nearestDistance = Number.POSITIVE_INFINITY;
      anchor.points.forEach((point, index) => {
        const step = point.step ?? index;
        const distance = Math.abs(step - hoveredStep);
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestIndex = index;
        }
      });
      chart.dispatchAction({ type: 'showTip', seriesIndex: plottedSeries.indexOf(anchor), dataIndex: nearestIndex });
    } finally {
      applyingSharedHoverRef.current = false;
    }
  }, [hoveredStep, plottedSeries]);

  useEffect(() => {
    if (!onHoverStep) return;
    const clearWhenPointerLeavesChart = (event: MouseEvent) => {
      const element = elementRef.current;
      if (!pointerInsideRef.current || !element || element.contains(event.target as Node)) return;
      pointerInsideRef.current = false;
      onHoverStep(null);
    };
    document.addEventListener('mousemove', clearWhenPointerLeavesChart, { passive: true });
    return () => document.removeEventListener('mousemove', clearWhenPointerLeavesChart);
  }, [onHoverStep]);

  return (
    <div>
      <div
        ref={elementRef}
        style={{ height: renderedHeight }}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
        onMouseEnter={() => { pointerInsideRef.current = true; }}
        onMouseLeave={() => {
          pointerInsideRef.current = false;
          onHoverStep?.(null);
        }}
      />
      <div className="sr-only">
        <p>{ariaLabel}. Values are plotted by {xDomain === 'elapsed-time' ? 'elapsed run time' : 'logical step'}.</p>
        <ul>
          {series.map((item) => {
            const latest = item.points.at(-1);
            return (
              <li key={item.name}>
                {item.name}: {item.points.length} points; latest value {latest?.value ?? 'unavailable'}
                {latest?.step == null ? '' : ` at step ${latest.step}`}.
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
