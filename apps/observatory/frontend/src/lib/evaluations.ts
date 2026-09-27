import type { EvaluationIndex, EvaluationRecord, EvaluationTaskScore } from './api';

/**
 * Evaluation runs grouped by the model they scored. A checkpoint evaluation
 * names its training run and step (parent_run, parent_step); a base-model
 * evaluation names neither. A suite is the evaluation's work package: scores
 * are compared only inside one suite.
 */

/** A score that can be compared: the run finished and recorded one. */
export function usable(record: EvaluationRecord): boolean {
  return record.status !== 'failed' && record.score != null;
}

export function suiteLabel(suite: string | null | undefined): string {
  if (!suite) return 'Unassigned suite';
  const parts = suite.split('/');
  return parts[parts.length - 1] || suite;
}

export function stepLabel(step: number | null | undefined): string {
  return step == null ? 'step —' : `step ${step}`;
}

function byStepThenStart(left: EvaluationRecord, right: EvaluationRecord): number {
  return (left.parent_step ?? -1) - (right.parent_step ?? -1)
    || (left.started_at ?? '').localeCompare(right.started_at ?? '');
}

/** Checkpoint evaluations of each training run, ordered by step. */
export function evaluationsByParent(index: EvaluationIndex | null): Map<string, EvaluationRecord[]> {
  const byParent = new Map<string, EvaluationRecord[]>();
  for (const record of index?.records ?? []) {
    if (!record.parent_run) continue;
    const list = byParent.get(record.parent_run) ?? [];
    list.push(record);
    byParent.set(record.parent_run, list);
  }
  for (const list of byParent.values()) list.sort(byStepThenStart);
  return byParent;
}

/** Base-model evaluations of ``model`` on ``suite``, oldest first. */
export function baseEvaluations(index: EvaluationIndex | null, suite: string | null | undefined, model: string | null | undefined): EvaluationRecord[] {
  return (index?.records ?? [])
    .filter((record) => !record.parent_run && record.suite === suite && record.model === model)
    .sort(byStepThenStart);
}

export function meanScore(records: EvaluationRecord[]): number | null {
  const scores = records.filter(usable).map((record) => record.score as number);
  return scores.length ? scores.reduce((sum, value) => sum + value, 0) / scores.length : null;
}

export type SuiteEvaluations = {
  suite: string | null;
  model: string | null;
  base: EvaluationRecord[];
  baseScore: number | null;
  checkpoints: EvaluationRecord[];
};

/** A training run's checkpoint evaluations split by suite, each with its base-model reference. */
export function runSuites(index: EvaluationIndex | null, runId: string): SuiteEvaluations[] {
  const checkpoints = evaluationsByParent(index).get(runId) ?? [];
  const groups = new Map<string, SuiteEvaluations>();
  for (const record of checkpoints) {
    const key = `${record.suite ?? ''}\u0000${record.model ?? ''}`;
    let group = groups.get(key);
    if (!group) {
      const base = baseEvaluations(index, record.suite, record.model);
      group = { suite: record.suite ?? null, model: record.model ?? null, base, baseScore: meanScore(base), checkpoints: [] };
      groups.set(key, group);
    }
    group.checkpoints.push(record);
  }
  return [...groups.values()].sort((left, right) => suiteLabel(left.suite).localeCompare(suiteLabel(right.suite)));
}

/** The latest usable checkpoint evaluation, else the latest one. */
export function defaultCheckpoint(records: EvaluationRecord[]): EvaluationRecord | null {
  const usableRecords = records.filter(usable);
  return usableRecords[usableRecords.length - 1] ?? records[records.length - 1] ?? null;
}

export type TrainingRunEvaluations = {
  runId: string;
  runKey: string;
  model: string | null;
  suites: string[];
};

/** Training runs with at least one checkpoint evaluation, and the suites they were scored on. */
export function evaluatedTrainingRuns(index: EvaluationIndex | null): TrainingRunEvaluations[] {
  return [...evaluationsByParent(index).entries()].map(([runId, records]) => ({
    runId,
    runKey: records[0].parent_run_key ?? '',
    model: records[0].model ?? null,
    suites: [...new Set(records.map((record) => record.suite ?? ''))].sort(),
  })).sort((left, right) => left.runId.localeCompare(right.runId));
}

export type TaskRow = { task: string; values: Record<string, number | null> };

/**
 * One row per task, one value per column. A column averages the task scores
 * of its runs (several base runs become one base column).
 */
export function taskRows(scores: EvaluationTaskScore[], columns: Array<{ key: string; runKeys: string[] }>): TaskRow[] {
  const byRunTask = new Map(scores.map((score) => [`${score.run_key}\u0000${score.task}`, score.score ?? null]));
  const tasks = [...new Set(scores.map((score) => score.task))].sort();
  return tasks.map((task) => ({
    task,
    values: Object.fromEntries(columns.map(({ key, runKeys }) => {
      const values = runKeys.map((runKey) => byRunTask.get(`${runKey}\u0000${task}`)).filter((value): value is number => value != null);
      return [key, values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null];
    })),
  }));
}

export function formatScore(value: number | null | undefined): string {
  return value == null ? '—' : value.toFixed(3);
}

export function formatDelta(value: number | null | undefined): string {
  if (value == null) return '—';
  const rounded = Number(value.toFixed(3));
  return `${rounded > 0 ? '+' : rounded < 0 ? '−' : '±'}${Math.abs(rounded).toFixed(3)}`;
}

export function deltaClass(value: number | null | undefined): string {
  if (value == null || Math.abs(value) < 0.0005) return 'text-muted';
  return value > 0 ? 'text-emerald-700' : 'text-rose-700';
}
