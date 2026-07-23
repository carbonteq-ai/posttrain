import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, LineChart, ScatterChart } from 'echarts/charts';
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
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
  TooltipComponent,
  CanvasRenderer,
]);

const colors = ['#6356c7', '#e85d3f', '#148c87', '#c68a2c', '#2476a8', '#8f548f'];

function shortName(name: string, metricLabels: Record<string, string>): string {
  return metricLabels[name] ?? name.split('/').at(-1)?.replaceAll('_', ' ') ?? name;
}

function axisGroup(name: string): string {
  if (name === 'train/loss' || name === 'train/validation/loss') return 'loss';
  if (name === 'train/mean_token_accuracy' || name === 'train/gradient_clipped' || name === 'train/rewards/accuracies') return 'ratio';
  if (name.startsWith('train/rewards/') && name !== 'train/rewards/accuracies') return 'dpo-reward';
  if (name.startsWith('train/logps/')) return 'dpo-log-probability';
  if (name.startsWith('train/logits/')) return 'dpo-logit';
  if (name === 'train/learning_rate') return 'learning-rate';
  if (name === 'train/non_padding_tokens_per_second') return 'tokens-per-second';
  if (name === 'train/step_time_seconds') return 'seconds';
  return name;
}

type EvidenceChartProps = {
  series: MetricSeries[];
  height?: number;
  compact?: boolean;
  ariaLabel: string;
  metricLabels?: Record<string, string>;
  showLegend?: boolean;
  selectedStep?: number | null;
  onPointSelect?: (step: number) => void;
};

export function EvidenceChart({
  series,
  height = 330,
  compact = false,
  ariaLabel,
  metricLabels = {},
  showLegend = true,
  selectedStep,
  onPointSelect,
}: EvidenceChartProps) {
  const elementRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!elementRef.current) return;
    const chart = echarts.init(elementRef.current, undefined, { renderer: 'canvas' });
    const pointCount = Math.max(0, ...series.map((item) => item.points.length));
    const showZoom = !compact && pointCount > 24;
    const axisGroups = [...new Set(series.map((item) => axisGroup(item.name)))].slice(0, 3);
    const axes = axisGroups.map((group, index) => ({
      type: 'value' as const,
      name: '',
      position: index === 0 ? ('left' as const) : ('right' as const),
      offset: index > 1 ? 52 : 0,
      scale: true,
      axisLine: { show: index > 0, lineStyle: { color: colors[index] } },
      axisLabel: {
        color: '#817a83',
        fontSize: 10,
        formatter: group.includes('bytes')
          ? (value: number) => `${(value / 1024 ** 3).toFixed(1)} GiB`
          : group === 'ratio'
            ? (value: number) => `${(value * 100).toFixed(0)}%`
            : undefined,
      },
      splitLine: {
        show: index === 0,
        lineStyle: { color: '#e8e4de', type: 'dashed' as const },
      },
    }));
    chart.setOption({
      animation: false,
      color: colors,
      grid: {
        left: compact ? 46 : 56,
        right: axes.length > 2 ? 92 : axes.length > 1 ? 54 : 24,
        top: compact ? 32 : 44,
        bottom: compact ? 34 : showZoom ? 72 : 46,
      },
      tooltip: { trigger: 'axis', confine: true },
      legend: {
        show: showLegend && series.length > 1,
        type: 'scroll',
        top: 4,
        left: 0,
        itemWidth: 16,
        itemHeight: 2,
        itemGap: 18,
        textStyle: { color: '#5f5a62', fontSize: 10 },
        formatter: (name: string) => shortName(name, metricLabels),
      },
      xAxis: {
        type: 'value',
        name: compact ? '' : 'Logical step',
        nameLocation: 'middle',
        nameGap: 34,
        axisLine: { lineStyle: { color: '#aaa4ab' } },
        axisTick: { show: false },
        axisLabel: { color: '#78727a', fontSize: 11 },
        splitLine: { show: false },
      },
      yAxis: axes.length ? axes : [{ type: 'value' }],
      dataZoom: compact || !showZoom
        ? []
        : [
            { type: 'inside', xAxisIndex: 0 },
            {
              type: 'slider',
              xAxisIndex: 0,
              height: 18,
              bottom: 10,
              borderColor: '#d8d3cb',
              backgroundColor: '#f7f5f1',
              fillerColor: 'rgba(99, 86, 199, .10)',
              handleStyle: { color: '#ffffff', borderColor: '#6356c7' },
              textStyle: { color: '#78727a', fontSize: 9 },
            },
          ],
      series: series.map((item) => ({
        name: item.name,
        type: 'line',
        yAxisIndex: Math.max(0, axisGroups.indexOf(axisGroup(item.name))),
        showSymbol: item.points.length < 12,
        symbolSize: 5,
        smooth: false,
        lineStyle: { width: 1.9 },
        emphasis: { focus: 'series' },
        markLine: selectedStep == null ? undefined : {
          silent: true,
          symbol: 'none',
          label: { show: false },
          lineStyle: { color: '#716a73', type: 'dashed', width: 1 },
          data: [{ xAxis: selectedStep }],
        },
        data: item.points.map((point, pointIndex) => [point.step ?? pointIndex, point.value]),
      })),
    });
    if (onPointSelect) {
      chart.on('click', (event) => {
        const data = event.data;
        if (Array.isArray(data) && typeof data[0] === 'number') onPointSelect(data[0]);
      });
    }
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(elementRef.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [compact, metricLabels, onPointSelect, selectedStep, series, showLegend]);

  return (
    <div>
      <div
        ref={elementRef}
        style={{ height }}
        className="w-full"
        role="img"
        aria-label={ariaLabel}
      />
      <div className="sr-only">
        <p>{ariaLabel}. Values are plotted by logical step.</p>
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
