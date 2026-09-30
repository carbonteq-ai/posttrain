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

/**
 * Short suite names that stay distinct: the words every suite name shares at
 * the start (``automationbench-``) are dropped, so a narrow list shows what
 * differs (``heldout-matched-64k-v3`` against ``vortex-matched-48k-v2``).
 */
export function shortSuiteLabels(suites: Array<string | null | undefined>): Map<string, string> {
  const labels = [...new Set(suites.map((suite) => suiteLabel(suite)))];
  const words = labels.map((label) => label.split('-'));
  let shared = 0;
  while (words.length > 1 && words.every((parts) => parts.length > shared + 1 && parts[shared] === words[0][shared])) shared += 1;
  return new Map(suites.map((suite) => [suite ?? '', suiteLabel(suite).split('-').slice(shared).join('-')]));
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

/** Per-episode behaviour recorded beside the score of every evaluation run. */
export type BehaviourKey = 'turns' | 'turns_completed' | 'tool_calls' | 'output_tokens' | 'thinking_tokens';

export const BEHAVIOUR_COLUMNS: ReadonlyArray<{ key: BehaviourKey; label: string; title: string; digits: number }> = [
  { key: 'turns', label: 'Turns', title: 'Mean model calls (turns) per episode', digits: 1 },
  { key: 'turns_completed', label: 'Turns (completed)', title: 'Mean turns on episodes that ended on their own', digits: 1 },
  { key: 'tool_calls', label: 'Tool calls', title: 'Mean tool calls per episode', digits: 1 },
  { key: 'output_tokens', label: 'Output tokens', title: 'Mean tokens generated per episode, thinking included', digits: 0 },
  { key: 'thinking_tokens', label: 'Thinking tokens', title: 'Mean thinking tokens per episode', digits: 0 },
];

export const NOT_RECORDED = 'Not recorded for this run';

const NUMBER_FORMATS = new Map<number, Intl.NumberFormat>();

export function formatMean(value: number | null | undefined, digits: number): string {
  if (value == null) return '—';
  let format = NUMBER_FORMATS.get(digits);
  if (!format) {
    format = new Intl.NumberFormat('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    NUMBER_FORMATS.set(digits, format);
  }
  return format.format(value);
}

/** Mean of one behaviour value over the records that recorded it; null when none did. */
export function behaviourMean(records: readonly EvaluationRecord[], key: BehaviourKey): number | null {
  const values = records.map((record) => record[key]).filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

/** How a run's episodes ended, most common first: "completed 97 · turn limit 3". */
export function endingSummary(endings: Record<string, number> | null | undefined): string {
  return Object.entries(endings ?? {})
    .filter(([, count]) => count > 0)
    .sort(([nameA, countA], [nameB, countB]) => countB - countA || nameA.localeCompare(nameB))
    .map(([name, count]) => `${name.replace(/_/g, ' ')} ${count}`)
    .join(' · ');
}
