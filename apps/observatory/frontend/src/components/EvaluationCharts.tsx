import { useEffect, useMemo, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, ScatterChart } from 'echarts/charts';
import { GridComponent, TooltipComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import type { TraceEvaluation } from '../lib/api';

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

export function EvaluationCharts({ evaluation }: { evaluation: TraceEvaluation }) {
  const options = useMemo(() => {
    const rewards = evaluation.traces.flatMap((trace) => (trace.reward == null ? [] : [trace.reward]));
    const minimum = rewards.length ? Math.floor(Math.min(...rewards) * 5) / 5 : 0;
    const maximum = rewards.length ? Math.ceil(Math.max(...rewards) * 5) / 5 : 1;
    const bins = Array.from({ length: 8 }, (_, index) => {
      const start = minimum + ((maximum - minimum || 1) / 8) * index;
      const end = minimum + ((maximum - minimum || 1) / 8) * (index + 1);
      return {
        label: start.toFixed(1),
        count: rewards.filter((reward) => reward >= start && (index === 7 ? reward <= end : reward < end)).length,
      };
    });
    return {
      rewards: {
        grid: { left: 35, right: 12, top: 12, bottom: 28 },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'category', data: bins.map((bin) => bin.label), ...axis },
        yAxis: { type: 'value', ...axis },
        series: [{ type: 'bar', data: bins.map((bin) => bin.count), itemStyle: { color: '#5bbcb7' }, barWidth: '82%' }],
      },
      slices: {
        grid: { left: 82, right: 28, top: 12, bottom: 28 },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value', min: 0, max: 100, ...axis },
        yAxis: { type: 'category', data: evaluation.slices.map((slice) => slice.key), ...axis },
        series: [{
          type: 'bar',
          data: evaluation.slices.map((slice) => Math.round((slice.success_rate ?? 0) * 1000) / 10),
          itemStyle: { color: '#5bbcb7' },
          label: { show: true, position: 'right', formatter: '{c}%', color: '#5f5961', fontSize: 11 },
        }],
      },
      tools: {
        grid: { left: 40, right: 16, top: 12, bottom: 28 },
        tooltip: { trigger: 'item' },
        xAxis: { type: 'value', name: 'Tool calls', nameLocation: 'middle', nameGap: 22, ...axis },
        yAxis: { type: 'value', name: 'Reward', ...axis },
        series: [
          {
            name: 'Pass',
            type: 'scatter',
            symbolSize: 7,
            itemStyle: { color: '#12a5a1' },
            data: evaluation.traces.filter((trace) => trace.success).map((trace) => [trace.tool_calls ?? 0, trace.reward ?? 0]),
          },
          {
            name: 'Review',
            type: 'scatter',
            symbolSize: 7,
            itemStyle: { color: '#e55f4a' },
            data: evaluation.traces.filter((trace) => !trace.success).map((trace) => [trace.tool_calls ?? 0, trace.reward ?? 0]),
          },
        ],
      },
    };
  }, [evaluation]);

  return (
    <div className="obs-card grid overflow-hidden divide-x divide-divider lg:grid-cols-3">
      <section className="p-4">
        <h3 className="text-[13px] font-medium">Reward distribution</h3>
        <p className="mt-1 text-xs text-muted">{evaluation.included} trace records</p>
        <MiniChart option={options.rewards} label="Histogram of trace rewards" />
      </section>
      <section className="p-4">
        <h3 className="text-[13px] font-medium">Pass rate by task slice</h3>
        <p className="mt-1 text-xs text-muted">Derived from verifier outcomes</p>
        <MiniChart option={options.slices} label="Pass rate by evaluation slice" />
      </section>
      <section className="p-4">
        <h3 className="text-[13px] font-medium">Reward vs. tool calls</h3>
        <p className="mt-1 text-xs text-muted">Pass and review populations</p>
        <MiniChart option={options.tools} label="Trace rewards plotted against tool calls" />
      </section>
    </div>
  );
}
