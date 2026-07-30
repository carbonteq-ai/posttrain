import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import type { SortingState } from '@tanstack/react-table';
import * as Popover from '@radix-ui/react-popover';
import {
  ArrowRight,
  ArrowClockwise,
  ArrowSquareOut,
  Bell,
  CaretDown,
  CaretRight,
  Check,
  Circle,
  CircleNotch,
  Cpu,
  Database,
  FileText,
  FolderSimple,
  GitDiff,
  Info,
  MagnifyingGlass,
  Planet,
  Pulse,
  SlidersHorizontal,
  Stack,
  TreeStructure,
  Warning,
  X,
} from '@phosphor-icons/react';

import { FilterPopover } from './components/FilterPopover';
import { PhaseMemoryTimeline } from './components/PhaseMemoryTimeline';
import { ServingBenchmarkOverview } from './features/serving/ServingBenchmarkOverview';
import { ServingCapacityWorkPackageView } from './features/serving/ServingCapacityWorkPackage';

import {
  api,
  type Artifact,
  type MetricHelp,
  type MetricSeries,
  type RunItem,
  type RunView,
  type SourceRefreshStatus,
  type ServingCapacityWorkPackage,
  type SystemMetrics,
  type TraceDetail,
  type TraceEvaluation,
  type TraceSummary,
  type WorkPackageView,
} from './lib/api';

const EvidenceChart = lazy(() =>
  import('./components/EvidenceChart').then((module) => ({ default: module.EvidenceChart })),
);
const EvaluationCharts = lazy(() =>
  import('./components/EvaluationCharts').then((module) => ({ default: module.EvaluationCharts })),
);
const TraceTable = lazy(() =>
  import('./components/TraceTable').then((module) => ({ default: module.TraceTable })),
);

type Section = 'Overview' | 'Metrics' | 'System metrics' | 'Traces & evaluation' | 'Artifacts & lineage' | 'Run config';

function sectionLabel(section: Section, jobKind: string): string {
  return section === 'Traces & evaluation' && jobKind === 'train.grpo'
    ? 'Rollouts & rewards'
    : section;
}

type SidebarWorkPackage = {
  id: string;
  stage: string;
  runs: RunItem[];
  latestStartedAt: string;
};

type SidebarStage = {
  stage: string;
  packages: SidebarWorkPackage[];
};

type SourceOption = {
  sourceId: string;
  provider: string;
  runCount: number;
};

const workflowStageOrder = ['screen', 'train', 'qualify'];

const sections: Section[] = [
  'Overview',
  'Metrics',
  'System metrics',
  'Traces & evaluation',
  'Artifacts & lineage',
  'Run config',
];

const jobCopy: Record<string, { eyebrow: string; title: string; question: string }> = {
  'train.sft': {
    eyebrow: 'SUPERVISED FINE-TUNING',
    title: 'Learning & data evidence',
    question: 'Is held-out loss improving without unstable updates, damaged supervision, or falling token throughput?',
  },
  'train.dpo': {
    eyebrow: 'DIRECT PREFERENCE OPTIMIZATION',
    title: 'Preference learning evidence',
    question: 'Is the policy learning the intended ordering without weakening chosen responses or destabilizing updates?',
  },
  'train.grpo': {
    eyebrow: 'GROUP RELATIVE POLICY OPTIMIZATION',
    title: 'Policy learning evidence',
    question: 'Does reward improve while rollout coverage, update stability, policy freshness, and runtime efficiency remain healthy?',
  },
  'train.sampo': {
    eyebrow: 'STEP-AWARE MULTI-TURN POLICY OPTIMIZATION',
    title: 'Hierarchical policy-learning evidence',
    question: 'Do episode and turn-level credit assignment improve multi-turn behavior without destabilizing policy updates?',
  },
  'train.distill': {
    eyebrow: 'ON-POLICY DISTILLATION',
    title: 'Student learning evidence',
    question: 'Is the student matching fresh teacher scores while retaining complete rollout and teacher-runtime evidence?',
  },
  'data.prepare': {
    eyebrow: 'DATASET PREPARATION',
    title: 'Prepared dataset evidence',
    question: 'Did the selected source produce the expected immutable dataset population and retained artifact?',
  },
  'eval.domain': {
    eyebrow: 'DOMAIN EVALUATION',
    title: 'Evaluation evidence',
    question: 'Which task slices and exact traces explain aggregate quality?',
  },
  'serve.benchmark': {
    eyebrow: 'SERVING CAPACITY',
    title: 'Serving benchmark',
    question: 'Does this model and serving configuration satisfy the product envelope on the fixed hardware profile?',
  },
  'serve.smoke': {
    eyebrow: 'SERVING HEALTH',
    title: 'Managed endpoint smoke test',
    question: 'Did the endpoint start successfully, answer its health check, and expose the selected model?',
  },
};

function formatValue(value: unknown, unit?: string | null): string {
  if (value == null || typeof value !== 'number') return '—';
  if (unit === 'bytes') {
    const gib = value / 1024 ** 3;
    return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
  }
  if (unit === 'ratio') return `${(value * 100).toFixed(1)}%`;
  if (unit === '%') return `${value.toFixed(0)}%`;
  if (unit === 's') return value >= 60 ? `${Math.floor(value / 60)}m ${Math.round(value % 60)}s` : `${value.toFixed(value >= 10 ? 1 : 2)}s`;
  if (unit === 'GiB') return `${value.toFixed(value >= 10 ? 1 : 2)} GiB`;
  if (unit === 'samples/s') return `${value.toFixed(2)} samples/s`;
  if (unit === 'tokens/s') return `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })} tokens/s`;
  if (unit === 'tokens') return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} tokens`;
  if (Math.abs(value) < 0.01 && value !== 0) return value.toExponential(2);
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'In progress';
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium',
  }).format(new Date(value));
}

function formatSidebarTimestamp(value: string): string {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value));
}

function timestampValue(value: string): number {
  const timestamp = Date.parse(value);
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

const metricLabels: Record<string, string> = {
  'train/loss': 'Loss',
  'train/validation/loss': 'Validation loss',
  'train/learning_rate': 'Learning rate',
  'train/grad_norm': 'Gradient norm',
  'train/gradient_clipped': 'Gradient clipped',
  'train/mean_token_accuracy': 'Token accuracy',
  'train/non_padding_tokens_per_second': 'Non-padding tokens / second',
  'train/step_time_seconds': 'Step time',
  'train/rewards/margins': 'Reward margin',
  'train/rewards/chosen': 'Chosen reward',
  'train/rewards/rejected': 'Rejected reward',
};

const metricUnits: Record<string, string> = {
  'train/mean_token_accuracy': 'ratio',
  'train/gradient_clipped': 'ratio',
  'train/rewards/accuracies': 'ratio',
};

function metricLabel(name: string): string {
  return metricLabels[name] ?? name.split('/').at(-1)?.replaceAll('_', ' ') ?? name;
}

function artifactLabel(kind: string): string {
  return {
    'model-adapter': 'Trained adapter',
    'model-weights': 'Trained weights',
    'training-checkpoint': 'Recovery checkpoint',
    'training-summary': 'Training summary',
  }[kind] ?? kind.replaceAll('-', ' ');
}

type SelectionValue = {
  id: string;
  revision: string | null;
  detail: Record<string, unknown>;
};

type MethodParameter = {
  label: string;
  value: string;
};

type ConfigEntryValue = SelectionValue & {
  sourceLayer: string | null;
  overlayId: string | null;
};

type ConfigGroupDefinition = {
  title: string;
  description: string;
  keys: string[];
};

const selectionLabels: Record<string, string> = {
  recipe: 'Recipe',
  model: 'Model',
  student_model: 'Student model',
  teacher_model: 'Teacher model',
  dataset: 'Training dataset',
  validation_dataset: 'Validation dataset',
  environment: 'Environment',
  inference: 'Inference binding',
  teacher_inference: 'Teacher inference binding',
  evaluation: 'Evaluation plan',
  workload: 'Workload',
  target: 'Execution target',
  execution_target: 'Execution target',
  settings: 'Optimization settings',
  training: 'Training binding',
  job_definition: 'Job definition',
  work_package: 'Work package',
};

const evaluationConfigGroups: ConfigGroupDefinition[] = [
  { title: 'Evaluation inputs', description: 'The model, task population, environment, and evaluation policy used to produce evidence.', keys: ['model', 'dataset', 'environment', 'evaluation'] },
  { title: 'Execution', description: 'The inference and workload selections that governed evaluation execution.', keys: ['recipe', 'inference', 'workload', 'target', 'execution_target', 'settings'] },
  { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
];

const jobConfigGroups: Record<string, ConfigGroupDefinition[]> = {
  'train.sft': [
    { title: 'Training population', description: 'The model and the disjoint data partitions used to fit and monitor supervised learning.', keys: ['model', 'dataset', 'validation_dataset'] },
    { title: 'Optimization', description: 'The recipe, trainer settings, and concrete backend binding that produced this run.', keys: ['recipe', 'settings', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'train.dpo': [
    { title: 'Preference inputs', description: 'The starting policy and preference-pair populations used for optimization and validation.', keys: ['model', 'dataset', 'validation_dataset'] },
    { title: 'Preference optimization', description: 'The DPO objective, settings, and concrete training backend selected for this run.', keys: ['recipe', 'settings', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'train.grpo': [
    { title: 'Policy & task inputs', description: 'The policy, task population, and environment against which rollouts were produced.', keys: ['model', 'dataset', 'validation_dataset', 'environment'] },
    { title: 'Rollout & optimization', description: 'The inference, training, and runtime selections governing policy updates.', keys: ['recipe', 'inference', 'settings', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'train.distill': [
    { title: 'Student, teacher & data', description: 'The trainable student, scoring teacher, and task population used for distillation.', keys: ['student', 'student_model', 'teacher', 'teacher_model', 'dataset', 'validation_dataset'] },
    { title: 'Generation & scoring', description: 'The environment and inference selections used to generate and score student responses.', keys: ['environment', 'inference', 'teacher_inference'] },
    { title: 'Distillation optimization', description: 'The objective, settings, and training binding used to update the student.', keys: ['recipe', 'settings', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'train.sampo': [
    { title: 'Policy & environment', description: 'The policy and multi-turn environment used to generate step-aware trajectories.', keys: ['model', 'environment'] },
    { title: 'Hierarchical optimization', description: 'The SAMPO settings, rollout inference, and training binding used for credit assignment and policy updates.', keys: ['settings', 'rollout_inference', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'serve.smoke': [
    { title: 'Serving binding', description: 'The immutable inference selection launched for this health qualification.', keys: ['inference'] },
    { title: 'Execution', description: 'The target selected to host the managed endpoint.', keys: ['target', 'execution_target'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'data.prepare': [
    { title: 'Dataset input', description: 'The source dataset selection validated and canonicalized by this job.', keys: ['dataset'] },
    { title: 'Execution', description: 'The target selected to materialize the immutable dataset snapshot.', keys: ['target', 'execution_target'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ],
  'eval.general': evaluationConfigGroups,
  'eval.domain': evaluationConfigGroups,
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function humanizeKey(value: string): string {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function configEntry(inputs: Record<string, unknown>, key: string): ConfigEntryValue | null {
  const raw = inputs[key];
  if (!isRecord(raw)) return null;
  const resolved = isRecord(raw.resolved) ? raw.resolved : null;
  const detail = resolved ?? Object.fromEntries(
    Object.entries(raw).filter(([field]) => !['selection_id', 'id', 'revision', 'source_layer', 'overlay_id', 'ref'].includes(field)),
  );
  const directId = raw.selection_id ?? raw.id ?? detail.id ?? detail.work_package_id ?? detail.target_id;
  return {
    id: typeof directId === 'string' ? directId : selectionLabels[key] ?? humanizeKey(key),
    revision: typeof raw.revision === 'string' ? raw.revision : null,
    sourceLayer: typeof raw.source_layer === 'string' ? raw.source_layer : null,
    overlayId: typeof raw.overlay_id === 'string' ? raw.overlay_id : null,
    detail,
  };
}

function configGroups(jobKind: string, inputs: Record<string, unknown>): ConfigGroupDefinition[] {
  const base = jobConfigGroups[jobKind] ?? [
    { title: 'Model & data', description: 'The reusable model and data selections supplied to this job.', keys: ['model', 'dataset', 'validation_dataset', 'environment'] },
    { title: 'Execution', description: 'The recipe, settings, and backend bindings used to execute this job.', keys: ['recipe', 'inference', 'evaluation', 'workload', 'target', 'execution_target', 'settings', 'training'] },
    { title: 'Package context', description: 'The job definition and work-package identity that placed this run in the larger workflow.', keys: ['job_definition', 'work_package'] },
  ];
  const assigned = new Set(base.flatMap((group) => group.keys));
  const remaining = Object.keys(inputs).filter((key) => !assigned.has(key));
  return remaining.length
    ? [...base, { title: 'Additional selections', description: 'Recorded selections that are not yet part of this job kind’s curated layout.', keys: remaining }]
    : base;
}

function selectionValue(inputs: Record<string, unknown> | undefined, key: string): SelectionValue | null {
  const raw = inputs?.[key];
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
  const value = raw as Record<string, unknown>;
  const detail = value.resolved && typeof value.resolved === 'object' && !Array.isArray(value.resolved)
    ? value.resolved as Record<string, unknown>
    : {};
  const id = value.selection_id ?? value.id ?? detail.id;
  if (typeof id !== 'string') return null;
  return {
    id,
    revision: typeof value.revision === 'string' ? value.revision : null,
    detail,
  };
}

function nestedValue(selection: SelectionValue | null, ...path: string[]): unknown {
  let value: unknown = selection?.detail;
  for (const key of path) {
    if (!isRecord(value)) return null;
    value = value[key];
  }
  return value;
}

function methodValue(value: unknown, format: 'plain' | 'upper' | 'tokens' | 'examples' | 'enabled' = 'plain'): string | null {
  if (value == null) return null;
  if (format === 'enabled' && typeof value === 'boolean') return value ? 'Enabled' : 'Disabled';
  if (format === 'upper' && typeof value === 'string') return value.toUpperCase();
  if (format === 'tokens' && typeof value === 'number') return `${value.toLocaleString()} tokens`;
  if (format === 'examples' && typeof value === 'number') return `${value.toLocaleString()} ${value === 1 ? 'example' : 'examples'}`;
  if (typeof value === 'number') return formatValue(value);
  if (typeof value === 'string') return humanizeKey(value);
  return null;
}

function methodParameters(
  jobKind: string,
  dataset: SelectionValue | null,
  validationDataset: SelectionValue | null,
  settings: SelectionValue | null,
  training: SelectionValue | null,
): MethodParameter[] {
  const parameterUpdate = nestedValue(training, 'parameter_update');
  const update = nestedValue(training, 'parameter_update_kind');
  const common = {
    update: methodValue(update, 'upper'),
    rank: methodValue(isRecord(parameterUpdate) ? parameterUpdate.rank : null),
    alpha: methodValue(isRecord(parameterUpdate) ? parameterUpdate.alpha : null),
    quantization: methodValue(isRecord(parameterUpdate) ? parameterUpdate.quant_type : null, 'upper'),
    learningRate: methodValue(nestedValue(settings, 'learning_rate')),
    maxSteps: methodValue(nestedValue(settings, 'max_steps')),
    maxLength: methodValue(nestedValue(settings, 'max_length'), 'tokens'),
    deviceBatch: methodValue(nestedValue(settings, 'per_device_batch_size')),
    globalBatch: methodValue(nestedValue(training, 'runtime', 'global_batch_size')),
    kernel: nestedValue(settings, 'loss_kernel') != null
      ? methodValue(nestedValue(settings, 'loss_kernel'))
      : nestedValue(training, 'runtime', 'use_liger_kernel') === true
        ? 'Liger'
        : nestedValue(training, 'runtime', 'use_liger_kernel') === false
          ? 'Standard'
          : null,
  };
  const definition: Array<[string, string | null]> = jobKind === 'train.dpo'
    ? [
        ['Update', common.update],
        ['Beta', methodValue(nestedValue(settings, 'beta'))],
        ['Loss kernel', common.kernel],
        ['Preference data', methodValue(nestedValue(dataset, 'num_examples'), 'examples')],
        ['Learning rate', common.learningRate],
        ['Context length', common.maxLength],
        ['Steps', common.maxSteps],
        ['Global batch', common.globalBatch],
      ]
    : jobKind === 'train.grpo'
      ? [
          ['Update', common.update],
          ['Group size', methodValue(nestedValue(settings, 'num_generations'))],
          ['KL beta', methodValue(nestedValue(settings, 'beta'))],
          ['Prompt limit', methodValue(nestedValue(settings, 'max_prompt_length'), 'tokens')],
          ['Completion limit', methodValue(nestedValue(settings, 'max_completion_length'), 'tokens')],
          ['Importance sampling', methodValue(nestedValue(settings, 'importance_sampling_mode'))],
          ['Learning rate', common.learningRate],
          ['Global batch', common.globalBatch],
        ]
      : jobKind === 'train.sampo'
        ? [
            ['Update', common.update],
            ['Group size', methodValue(nestedValue(settings, 'num_generations'))],
            ['Discount gamma', methodValue(nestedValue(settings, 'discount_gamma'))],
            ['Turn weight', methodValue(nestedValue(settings, 'step_advantage_weight'))],
            ['Advantage normalization', methodValue(nestedValue(settings, 'advantage_normalization'))],
            ['Clip range', [
              methodValue(nestedValue(settings, 'clip_epsilon_low')),
              methodValue(nestedValue(settings, 'clip_epsilon_high')),
            ].filter(Boolean).join(' – ') || null],
            ['Learning rate', common.learningRate],
            ['Global batch', common.globalBatch],
          ]
      : jobKind === 'train.distill'
        ? [
            ['Update', common.update],
            ['Objective', methodValue(nestedValue(settings, 'divergence'))],
            ['Temperature', methodValue(nestedValue(settings, 'temperature'))],
            ['Samples / prompt', methodValue(nestedValue(settings, 'num_generations'))],
            ['Prompt limit', methodValue(nestedValue(settings, 'max_prompt_length'), 'tokens')],
            ['Completion limit', methodValue(nestedValue(settings, 'max_completion_length'), 'tokens')],
            ['Learning rate', common.learningRate],
            ['Global batch', common.globalBatch],
          ]
        : jobKind === 'serve.smoke' || jobKind === 'data.prepare'
          ? []
        : [
            ['Update', common.update],
            ['Adapter rank', common.rank],
            ['Adapter alpha', common.alpha],
            ['Quantization', common.quantization],
            ['Training data', methodValue(nestedValue(dataset, 'num_examples'), 'examples')],
            ['Validation data', methodValue(nestedValue(validationDataset, 'num_examples'), 'examples')],
            ['Learning rate', common.learningRate],
            ['Context length', common.maxLength],
            ['Steps', common.maxSteps],
            ['Device batch', common.deviceBatch],
            ['Global batch', common.globalBatch],
            ['Kernel', common.kernel],
          ];
  return definition
    .filter((item): item is [string, string] => item[1] !== null)
    .slice(0, 8)
    .map(([label, value]) => ({ label, value }));
}

function Status({ value }: { value: string }) {
  const color = value === 'succeeded' || value === 'complete' ? 'text-emerald-600' : value === 'failed' ? 'text-rose-600' : 'text-amber-600';
  return (
    <span className={`inline-flex items-center gap-1.5 text-[11px] font-medium leading-none capitalize ${color}`}>
      <Circle size={6} weight="fill" aria-hidden="true" /> {value}
    </span>
  );
}

function projectLabel(projectId: string): string {
  return projectId.replace(/^projects\//, '');
}

function providerLabel(provider: string): string {
  return provider === 'wandb' ? 'Weights & Biases' : provider === 'trackio' ? 'Trackio' : humanizeKey(provider);
}

function packageLabel(workPackageId: string): string {
  const parts = workPackageId.split('/');
  return parts.at(-1) || workPackageId;
}

function ProjectSelector({ projects, value, onChange }: { projects: string[]; value: string; onChange: (projectId: string) => void }) {
  const [open, setOpen] = useState(false);
  return <Popover.Root open={open} onOpenChange={setOpen}>
    <Popover.Trigger asChild>
      <button type="button" aria-label={`Project: ${projectLabel(value)}`} className="flex w-full items-center gap-2 rounded-[5px] border border-divider bg-surface px-2.5 py-2 text-left transition hover:border-[#cfc8d7] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:ring-offset-1">
        <FolderSimple size={16} className="shrink-0 text-violet-700" aria-hidden="true" />
        <span className="min-w-0 flex-1"><span className="type-label block">Project</span><strong title={value} className="mt-0.5 block truncate text-[11px] font-medium text-ink">{projectLabel(value)}</strong></span>
        <CaretDown size={12} className="shrink-0 text-muted" aria-hidden="true" />
      </button>
    </Popover.Trigger>
    <Popover.Portal>
      <Popover.Content sideOffset={6} align="start" className="z-50 w-[260px] rounded-md border border-divider bg-surface p-1.5 shadow-[0_12px_32px_rgba(40,35,44,.12)] outline-none">
        <p className="px-2 pb-1.5 pt-1 text-[10px] font-medium uppercase tracking-[.1em] text-muted">Active project</p>
        <div role="listbox" aria-label="Project" className="space-y-0.5">
          {projects.map((project) => {
            const active = project === value;
            return <button key={project} type="button" role="option" aria-selected={active} onClick={() => { onChange(project); setOpen(false); }} className={`flex w-full items-center justify-between gap-3 rounded px-2 py-2 text-left text-xs ${active ? 'bg-violet-50 text-violet-800' : 'text-secondary hover:bg-subtle hover:text-ink'}`}>
              <span className="min-w-0"><strong className="block truncate font-medium">{projectLabel(project)}</strong><small title={project} className="mt-0.5 block truncate text-[9px] text-muted">{project}</small></span>
              {active && <Check size={13} weight="bold" className="shrink-0" aria-hidden="true" />}
            </button>;
          })}
        </div>
        <Popover.Arrow className="fill-surface" />
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}

function SourceSelector({
  sources,
  value,
  onChange,
  onRefresh,
  refreshStatus,
}: {
  sources: SourceOption[];
  value: string;
  onChange: (sourceId: string) => void;
  onRefresh: () => Promise<void>;
  refreshStatus: SourceRefreshStatus | null;
}) {
  const [open, setOpen] = useState(false);
  const active = sources.find((source) => source.sourceId === value) ?? sources[0];
  if (!active) return null;
  return <Popover.Root open={open} onOpenChange={setOpen}>
    <Popover.Trigger asChild>
      <button type="button" aria-label={`Backend: ${providerLabel(active.provider)}`} className="mb-2 flex w-full items-center gap-2 rounded-[5px] border border-divider bg-surface px-2.5 py-2 text-left transition hover:border-[#cfc8d7] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:ring-offset-1">
        <Database size={16} className="shrink-0 text-violet-700" aria-hidden="true" />
        <span className="min-w-0 flex-1"><span className="type-label block">Backend</span><strong title={active.sourceId} className="mt-0.5 block truncate text-[11px] font-medium text-ink">{providerLabel(active.provider)}</strong></span>
        <CaretDown size={12} className="shrink-0 text-muted" aria-hidden="true" />
      </button>
    </Popover.Trigger>
    <Popover.Portal>
      <Popover.Content sideOffset={6} align="start" className="z-50 w-[260px] rounded-md border border-divider bg-surface p-1.5 shadow-[0_12px_32px_rgba(40,35,44,.12)] outline-none">
        <div className="flex items-center justify-between">
          <p className="px-2 pb-1.5 pt-1 text-[10px] font-medium uppercase tracking-[.1em] text-muted">Evidence backend</p>
          <button
            type="button"
            aria-label="Refresh evidence backends"
            title="Refresh evidence backends"
            disabled={refreshStatus?.state === 'refreshing'}
            onClick={() => void onRefresh()}
            className="grid size-7 shrink-0 place-items-center rounded-md text-muted transition hover:text-ink disabled:cursor-wait disabled:opacity-60"
          >
            {refreshStatus?.state === 'refreshing'
              ? <CircleNotch size={13} className="animate-spin" aria-hidden="true" />
              : <ArrowClockwise size={13} aria-hidden="true" />}
          </button>
        </div>
        {refreshStatus?.state === 'failed' && refreshStatus.error && (
          <p role="alert" className="mx-2 mb-1.5 rounded bg-rose-50 px-2 py-1.5 text-[10px] leading-4 text-rose-700">
            {refreshStatus.error}
          </p>
        )}
        <div role="listbox" aria-label="Backend" className="space-y-0.5">
          {sources.map((source) => {
            const selected = source.sourceId === value;
            return <button key={source.sourceId} type="button" role="option" aria-selected={selected} onClick={() => { onChange(source.sourceId); setOpen(false); }} className={`flex w-full items-center justify-between gap-3 rounded px-2 py-2 text-left text-xs ${selected ? 'bg-violet-50 text-violet-800' : 'text-secondary hover:bg-subtle hover:text-ink'}`}>
              <span className="min-w-0"><strong className="block truncate font-medium">{providerLabel(source.provider)}</strong><small className="mt-0.5 block truncate text-[9px] text-muted">{source.sourceId} · {source.runCount} {source.runCount === 1 ? 'run' : 'runs'}</small></span>
              {selected && <Check size={13} weight="bold" className="shrink-0" aria-hidden="true" />}
            </button>;
          })}
        </div>
        <Popover.Arrow className="fill-surface" />
      </Popover.Content>
    </Popover.Portal>
  </Popover.Root>;
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="obs-card border-dashed bg-subtle px-8 py-14 text-center">
      <FileText className="mx-auto text-muted" size={26} aria-hidden="true" />
      <h2 className="mt-3 font-serif text-xl">{title}</h2>
      <p className="mx-auto mt-2 max-w-xl text-xs leading-5 text-muted">{body}</p>
    </div>
  );
}

function ChartFallback({ height = 250 }: { height?: number }) {
  return <div style={{ height }} className="grid place-items-center text-xs text-muted">Preparing evidence visualization…</div>;
}

function MetricInfo({ label, metric, help }: { label: string; metric: string; help?: MetricHelp }) {
  const content = help ?? {
    metric,
    label,
    description: 'A provider-recorded metric without registered job-specific semantics.',
    interpretation: 'Inspect the producing backend and run configuration before assigning meaning or comparing it across runs.',
    caveat: 'Observatory deliberately does not infer semantics for unregistered metrics.',
    unit: null,
  };
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button type="button" aria-label={`About ${label}`} className="-m-1 inline-grid size-6 shrink-0 place-items-center rounded-full text-muted transition hover:bg-violet-50 hover:text-violet-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-500 focus-visible:ring-offset-1">
          <Info size={14} aria-hidden="true" />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content role="dialog" aria-label={`${label} metric definition`} align="start" sideOffset={7} collisionPadding={12} className="z-50 w-[min(320px,calc(100vw-24px))] rounded-md border border-divider bg-surface p-4 text-left shadow-[0_14px_40px_rgba(40,32,48,.14)] outline-none">
          <div className="flex items-start justify-between gap-3">
            <div><p className="type-eyebrow">METRIC DEFINITION</p><h3 className="mt-1 font-serif text-lg leading-tight text-ink">{content.label}</h3></div>
            {content.unit && <span className="rounded border border-divider bg-subtle px-1.5 py-1 font-mono text-[9px] text-muted">{content.unit}</span>}
          </div>
          <p className="mt-3 text-xs leading-5 text-secondary">{content.description}</p>
          <div className="mt-3 border-t border-divider pt-3"><p className="type-label">HOW TO READ IT</p><p className="mt-1.5 text-xs leading-5 text-secondary">{content.interpretation}</p></div>
          {content.caveat && <div className="mt-3 rounded bg-subtle px-2.5 py-2 text-[11px] leading-4 text-muted"><strong className="font-medium text-secondary">Keep in mind:</strong> {content.caveat}</div>}
          <code className="mt-3 block break-all text-[9px] text-muted">{content.metric}</code>
          <Popover.Arrow className="fill-surface" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}

function MetricLabel({ label, metric, help, className = '' }: { label: string; metric: string | null; help?: MetricHelp; className?: string }) {
  return <span className={`inline-flex items-center gap-0.5 ${className}`}><span>{label}</span>{metric && <MetricInfo label={label} metric={metric} help={help} />}</span>;
}

function ContextValue({ label, value }: { label: string; value: string }) {
  return <div className="border-b border-r border-divider px-4 py-3"><span className="type-label block">{label}</span><strong className="mt-1 block text-xs font-medium text-ink">{value}</strong></div>;
}

export default function App() {
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [selected, setSelected] = useState<RunItem | null>(null);
  const [loadedView, setLoadedView] = useState<{ runKey: string; response: RunView } | null>(null);
  const [section, setSection] = useState<Section>('Overview');
  const [mode, setMode] = useState<'auto' | 'generic'>('auto');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [system, setSystem] = useState<SystemMetrics | null>(null);
  const [evaluation, setEvaluation] = useState<TraceEvaluation | null>(null);
  const [traceDetail, setTraceDetail] = useState<TraceDetail | null>(null);
  const [activeChart, setActiveChart] = useState(0);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [activeWorkPackageId, setActiveWorkPackageId] = useState<string | null>(null);
  const [workPackage, setWorkPackage] = useState<WorkPackageView | null>(null);
  const [servingCapacity, setServingCapacity] = useState<ServingCapacityWorkPackage | null>(null);
  const [workPackageLoading, setWorkPackageLoading] = useState(false);
  const [sourceRefreshStatus, setSourceRefreshStatus] = useState<SourceRefreshStatus | null>(null);
  const viewRequestSequence = useRef(0);
  const selectedRunKeyRef = useRef<string | null>(null);

  const loadView = useCallback(async (run: RunItem, nextMode: 'auto' | 'generic') => {
    const requestSequence = ++viewRequestSequence.current;
    setLoadedView(null);
    try {
      const value = await api.view(run.run_key, nextMode);
      if (requestSequence !== viewRequestSequence.current || selectedRunKeyRef.current !== run.run_key) return;
      setLoadedView({ runKey: run.run_key, response: value });
      setActiveChart(0);
    } catch (cause) {
      if (requestSequence === viewRequestSequence.current && selectedRunKeyRef.current === run.run_key) throw cause;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.runs()
      .then(async (items) => {
        if (cancelled) return;
        setRuns(items);
        if (items[0]) {
          selectedRunKeyRef.current = items[0].run_key;
          setSelected(items[0]);
          setSelectedSourceId(items[0].locator.source_id);
          setSelectedProject(items[0].run.project_id);
          await loadView(items[0], 'auto');
        }
      })
      .catch((cause: unknown) => { if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => {
      cancelled = true;
      selectedRunKeyRef.current = null;
      viewRequestSequence.current += 1;
    };
  }, [loadView]);

  const chooseRun = useCallback(async (run: RunItem) => {
    selectedRunKeyRef.current = run.run_key;
    setSelected(run);
    setSelectedSourceId(run.locator.source_id);
    setSelectedProject(run.run.project_id);
    setActiveWorkPackageId(null);
    setWorkPackage(null);
    setServingCapacity(null);
    setSection('Overview');
    setMode('auto');
    setSystem(null);
    setEvaluation(null);
    setTraceDetail(null);
    setLoadedView(null);
    setError('');
    try {
      await loadView(run, 'auto');
    } catch (cause) {
      if (selectedRunKeyRef.current === run.run_key) setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [loadView]);

  const chooseProject = useCallback(async (projectId: string) => {
    const run = runs.find((item) => item.locator.source_id === selectedSourceId && item.run.project_id === projectId);
    if (!run) return;
    setSearch('');
    setSelectedProject(projectId);
    await chooseRun(run);
  }, [chooseRun, runs, selectedSourceId]);

  const chooseSource = useCallback(async (sourceId: string) => {
    const run = runs.find((item) => item.locator.source_id === sourceId);
    if (!run) return;
    setSearch('');
    setSelectedSourceId(sourceId);
    await chooseRun(run);
  }, [chooseRun, runs]);

  const refreshSources = useCallback(async () => {
    setSourceRefreshStatus((current) => ({
      enabled: true,
      state: 'refreshing',
      last_attempt_at: current?.last_attempt_at ?? null,
      last_success_at: current?.last_success_at ?? null,
      error: null,
      discovered_source_ids: current?.discovered_source_ids ?? [],
    }));
    try {
      const status = await api.refreshSources();
      setSourceRefreshStatus(status);
      if (status.state === 'failed') return;
      const items = await api.runs();
      setRuns(items);
      const current = items.find((item) => item.run_key === selectedRunKeyRef.current);
      if (current) {
        setSelected(current);
        setSelectedSourceId(current.locator.source_id);
        setSelectedProject(current.run.project_id);
      } else if (items[0]) {
        await chooseRun(items[0]);
      } else {
        selectedRunKeyRef.current = null;
        setSelected(null);
        setSelectedSourceId(null);
        setSelectedProject(null);
      }
    } catch (cause) {
      setSourceRefreshStatus((current) => ({
        enabled: true,
        state: 'failed',
        last_attempt_at: current?.last_attempt_at ?? null,
        last_success_at: current?.last_success_at ?? null,
        error: cause instanceof Error ? cause.message : String(cause),
        discovered_source_ids: current?.discovered_source_ids ?? [],
      }));
    }
  }, [chooseRun]);

  const openWorkPackage = useCallback(async (projectId: string, workPackageId: string) => {
    setSelectedProject(projectId);
    setActiveWorkPackageId(workPackageId);
    setWorkPackage(null);
    setServingCapacity(null);
    setWorkPackageLoading(true);
    setError('');
    try {
      if (!selectedSourceId) throw new Error('No evidence backend is selected.');
      const [packageView, capacityView] = await Promise.all([
        api.workPackage(workPackageId, projectId, selectedSourceId),
        api.servingCapacity(workPackageId, projectId, selectedSourceId).catch(() => null),
      ]);
      setWorkPackage(packageView);
      setServingCapacity(capacityView);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setWorkPackageLoading(false);
    }
  }, [selectedSourceId]);

  const openSection = useCallback(async (next: Section) => {
    if (!selected) return;
    const runKey = selected.run_key;
    setSection(next);
    try {
      if (next === 'System metrics' && !system) {
        const value = await api.system(runKey);
        if (selectedRunKeyRef.current === runKey) setSystem(value);
      }
      if (next === 'Traces & evaluation' && !evaluation) {
        const value = await api.evaluation(runKey);
        if (selectedRunKeyRef.current !== runKey) return;
        setEvaluation(value);
        if (value.traces[0]) {
          const detail = await api.trace(runKey, value.traces[0].external_id);
          if (selectedRunKeyRef.current === runKey) setTraceDetail(detail);
        }
      }
      if (next === 'Metrics' && mode !== 'generic') {
        setMode('generic');
        await loadView(selected, 'generic');
      }
      if (next === 'Overview' && mode !== 'auto') {
        setMode('auto');
        await loadView(selected, 'auto');
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    }
  }, [evaluation, loadView, mode, selected, system]);

  const activeProject = selectedProject ?? selected?.run.project_id ?? '';
  const activeSourceId = selectedSourceId ?? selected?.locator.source_id ?? '';
  const sourceOptions = useMemo(() => {
    const values = new Map<string, SourceOption>();
    for (const run of runs) {
      const sourceId = run.locator.source_id;
      const current = values.get(sourceId);
      values.set(sourceId, {
        sourceId,
        provider: current?.provider ?? run.run.provider,
        runCount: (current?.runCount ?? 0) + 1,
      });
    }
    return [...values.values()].sort((left, right) => left.provider.localeCompare(right.provider));
  }, [runs]);
  const projects = useMemo(
    () => [...new Set(runs.filter((run) => run.locator.source_id === activeSourceId).map((run) => run.run.project_id))].sort(),
    [activeSourceId, runs],
  );
  const filteredRuns = useMemo(() => {
    const query = search.trim().toLowerCase();
    const scoped = runs.filter((run) => run.locator.source_id === activeSourceId && run.run.project_id === activeProject);
    if (!query) return scoped;
    return scoped.filter((run) =>
      [run.run.display_name, run.run.job_kind, run.run.work_package_id, run.run.provider, run.run.status]
        .join(' ')
        .toLowerCase()
        .includes(query),
    );
  }, [activeProject, activeSourceId, runs, search]);
  const sidebarPackages = useMemo(() => {
    const groups = new Map<string, RunItem[]>();
    for (const run of filteredRuns) {
      const values = groups.get(run.run.work_package_id) ?? [];
      values.push(run);
      groups.set(run.run.work_package_id, values);
    }
    return [...groups.entries()].map(([id, values]): SidebarWorkPackage => {
      const sortedRuns = [...values].sort((left, right) =>
        timestampValue(right.run.started_at) - timestampValue(left.run.started_at)
        || left.run.run_id.localeCompare(right.run.run_id));
      return {
        id,
        stage: sortedRuns[0]?.run.stage ?? id.split('/')[0] ?? 'work',
        runs: sortedRuns,
        latestStartedAt: sortedRuns[0]?.run.started_at ?? '',
      };
    });
  }, [filteredRuns]);
  const sidebarStages = useMemo(() => {
    const groups = new Map<string, SidebarWorkPackage[]>();
    for (const workPackage of sidebarPackages) {
      const values = groups.get(workPackage.stage) ?? [];
      values.push(workPackage);
      groups.set(workPackage.stage, values);
    }
    const orderedStages = [
      ...workflowStageOrder.filter((stage) => groups.has(stage)),
      ...[...groups.keys()].filter((stage) => !workflowStageOrder.includes(stage)).sort(),
    ];
    return orderedStages.map((stage): SidebarStage => ({
      stage,
      packages: [...(groups.get(stage) ?? [])].sort((left, right) =>
        timestampValue(right.latestStartedAt) - timestampValue(left.latestStartedAt)
        || left.id.localeCompare(right.id)),
    }));
  }, [sidebarPackages]);

  if (loading) return <div className="grid min-h-screen place-items-center bg-subtle text-sm text-muted">Loading evidence…</div>;
  if (!selected) return <div className="grid min-h-screen place-items-center bg-subtle text-sm text-rose-600">{error || 'No runs are available.'}</div>;
  const response = loadedView?.runKey === selected.run_key ? loadedView.response : null;

  const traceEvaluationEnabled = response?.view.run.run_id === selected.run.run_id
    && response.view.trace_evaluation_enabled;
  const visibleSections = sections.filter((item) => item !== 'Traces & evaluation' || traceEvaluationEnabled);

  const copy = jobCopy[selected.run.job_kind] ?? {
    eyebrow: selected.run.job_kind.toUpperCase(),
    title: response?.resolved_mode === 'generic' ? 'Generic evidence workspace' : 'Run evidence brief',
    question: 'Inspect recorded evidence without assuming undocumented metric meaning.',
  };

  return (
    <div data-theme="light" className="min-h-screen bg-canvas text-ink">
      <aside className="fixed inset-y-0 left-0 z-30 flex w-[244px] flex-col border-r border-divider bg-panel px-3 py-5">
        <div className="flex items-center gap-2 px-2 pb-4 font-serif text-xl">
          <Planet size={27} weight="thin" aria-hidden="true" /> Observatory
        </div>
        <SourceSelector
          sources={sourceOptions}
          value={activeSourceId}
          onChange={(sourceId) => void chooseSource(sourceId)}
          onRefresh={refreshSources}
          refreshStatus={sourceRefreshStatus}
        />
        <ProjectSelector projects={projects} value={activeProject} onChange={(projectId) => void chooseProject(projectId)} />
        <label className="obs-search mt-3 flex h-9 items-center gap-2 rounded border border-divider bg-surface px-2.5 text-muted focus-within:border-violet-500">
          <MagnifyingGlass size={15} aria-hidden="true" />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search project evidence"
            placeholder="Search this project"
            className="min-w-0 flex-1 bg-transparent text-xs outline-none focus-visible:outline-none"
          />
        </label>
        <nav aria-label="Primary" className="mt-3 flex gap-1 border-b border-divider pb-3 text-xs">
          {[
            [GitDiff, 'Compare'],
            [Bell, 'Alerts'],
          ].map(([Icon, label]) => (
            <button
              key={label as string}
              className="flex flex-1 items-center justify-center gap-1.5 rounded px-2 py-2 text-secondary hover:bg-subtle"
            >
              <Icon size={17} aria-hidden="true" /> {label as string}
            </button>
          ))}
        </nav>
        <div className="mt-4 min-h-0 flex-1 overflow-auto">
          <div className="flex items-center justify-between px-2"><p className="text-[10px] font-medium tracking-[.14em] text-muted">WORK PACKAGES</p><span className="rounded-full bg-subtle px-1.5 py-0.5 text-[9px] text-muted">{sidebarPackages.length}</span></div>
          <div className="mt-2 space-y-4">
            {sidebarStages.map((stageGroup) => (
              <section key={stageGroup.stage} aria-label={`${stageGroup.stage} work packages`}>
                <div className="flex items-center justify-between px-2 py-1"><p className="text-[9px] font-medium uppercase tracking-[.13em] text-muted">{stageGroup.stage}</p><span className="text-[9px] text-muted">{stageGroup.packages.length}</span></div>
                <div className="space-y-2">
                {stageGroup.packages.map((group) => <section key={group.id} aria-label={`Work package ${group.id}`}>
                <button type="button" aria-label={`Open work package ${group.id}`} onClick={() => void openWorkPackage(activeProject, group.id)} aria-current={activeWorkPackageId === group.id ? 'page' : undefined} className={`flex w-full items-start gap-2 rounded px-2 py-2 text-left ${activeWorkPackageId === group.id ? 'bg-violet-50 text-violet-800' : 'hover:bg-subtle'}`}>
                  <Stack size={15} className="mt-0.5 shrink-0" aria-hidden="true" />
                  <span className="min-w-0 flex-1"><strong title={group.id} className="block truncate text-[11px] font-medium leading-tight">{packageLabel(group.id)}</strong><small className="mt-1 block truncate text-[9px] text-muted">{group.runs.length} {group.runs.length === 1 ? 'run' : 'runs'} · latest {formatSidebarTimestamp(group.latestStartedAt)}</small></span>
                  <CaretRight size={11} className="mt-0.5 shrink-0 text-muted" aria-hidden="true" />
                </button>
                <div className="ml-[15px] mt-0.5 border-l border-divider pl-2">
                  {group.runs.map((run) => (
                    <button key={run.run_key} type="button" aria-label={`Select run ${run.run.display_name}`} onClick={() => void chooseRun(run)} className={`flex w-full items-start gap-2 rounded px-2 py-1.5 text-left ${!activeWorkPackageId && selected.run_key === run.run_key ? 'bg-violet-50 text-violet-800' : 'hover:bg-subtle'}`}>
                      <Circle className={`mt-1 shrink-0 ${run.run.status === 'succeeded' ? 'text-emerald-600' : run.run.status === 'failed' ? 'text-rose-600' : 'text-amber-600'}`} size={6} weight="fill" aria-hidden="true" />
                      <span className="min-w-0 flex-1"><strong title={run.run.display_name} className="block truncate text-[11px] font-medium leading-tight">{run.run.display_name}</strong><small className="mt-0.5 flex min-w-0 items-center justify-between gap-2 text-[9px] text-muted"><span className="truncate">{run.run.job_kind}</span><time className="shrink-0" dateTime={run.run.started_at} title={formatTimestamp(run.run.started_at)}>{formatSidebarTimestamp(run.run.started_at)}</time></small></span>
                    </button>
                  ))}
                </div>
              </section>)}
                </div>
              </section>
            ))}
            {!sidebarPackages.length && <p className="px-2 py-5 text-center text-[11px] leading-4 text-muted">No work packages match this project search.</p>}
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 border-t border-divider px-2 pt-4">
          <span className="grid size-8 place-items-center rounded-full bg-violet-100 text-xs text-violet-800">ML</span>
          <span><strong className="block text-xs">ML workspace</strong><small className="text-[11px] text-muted">Evidence reader</small></span>
        </div>
      </aside>

      <main className="ml-[244px] min-w-0 overflow-x-hidden">
        <header className="sticky top-0 z-20 flex h-[66px] items-center gap-4 border-b border-divider bg-panel/95 px-7 backdrop-blur">
          <Crumb label="Project" value={projectLabel(activeProject)} mobileGrow />
          <span className="hidden items-center gap-4 sm:contents">
            <CaretRight size={13} className="text-muted" aria-hidden="true" />
            <Crumb label="Work package" value={activeWorkPackageId ?? selected.run.work_package_id} grow={Boolean(activeWorkPackageId)} />
            {!activeWorkPackageId && <><CaretRight size={13} className="text-muted" aria-hidden="true" /><Crumb label="Run" value={selected.run.display_name} grow /></>}
          </span>
          <div className="flex items-center gap-5 text-xs">
            {activeWorkPackageId ? <span className="text-muted">{workPackageLoading ? 'Loading package…' : `${workPackage?.runs.length ?? 0} runs`}</span> : <><Status value={selected.run.status} /><span className="hidden text-muted sm:inline">{selected.run.provider}</span></>}
          </div>
        </header>
        {!activeWorkPackageId && <nav aria-label="Run evidence sections" className="flex h-9 gap-4 overflow-x-auto border-b border-divider bg-panel px-8">
          {visibleSections.map((item) => (
            <button
              key={item}
              onClick={() => void openSection(item)}
              className={`whitespace-nowrap border-b-2 pt-px text-[11px] tracking-[.01em] ${section === item ? 'border-violet-600 font-medium text-ink' : 'border-transparent text-muted hover:text-ink'}`}
            >
              {sectionLabel(item, selected.run.job_kind)}
            </button>
          ))}
        </nav>}

        {error && (
          <div className="mx-8 mt-4 flex items-center gap-2 border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
            <Warning size={16} weight="fill" aria-hidden="true" /> {error}
            <button className="ml-auto" aria-label="Dismiss error" onClick={() => setError('')}><X size={14} /></button>
          </div>
        )}

        <div className="mx-auto max-w-[1460px] px-8 py-7">
          {activeWorkPackageId ? <WorkPackagePage view={workPackage} servingCapacity={servingCapacity} loading={workPackageLoading} onOpenRun={(runKey) => {
            const run = runs.find((item) => item.run_key === runKey);
            if (run) void chooseRun(run);
          }} /> : !response ? <div className="grid min-h-[420px] place-items-center text-xs text-muted">{error || 'Loading run evidence…'}</div> : <>
          {section === 'Overview' && (
            response.view.view_kind === 'job.serving'
              ? <ServingBenchmarkOverview response={response} sourceId={selected.locator.source_id} onRunConfig={() => void openSection('Run config')} />
              : <Overview
                  selected={selected}
                  response={response}
                  copy={copy}
                  activeChart={activeChart}
                  onChart={setActiveChart}
                  onTraces={() => void openSection('Traces & evaluation')}
                />
          )}
          {section === 'Metrics' && <GenericMetrics response={response} runKey={selected.run_key} />}
          {section === 'System metrics' && <SystemView system={system} />}
          {section === 'Traces & evaluation' && (
            <TraceView
              jobKind={selected.run.job_kind}
              evaluation={evaluation}
              detail={traceDetail}
              onSelect={async (trace) => {
                const runKey = selected.run_key;
                const detail = await api.trace(runKey, trace.external_id);
                if (selectedRunKeyRef.current === runKey) setTraceDetail(detail);
              }}
            />
          )}
          {section === 'Artifacts & lineage' && <ArtifactsView response={response} />}
          {section === 'Run config' && <ConfigView response={response} />}
          </>}
        </div>
      </main>
    </div>
  );
}

function Crumb({ label, value, grow = false, mobileGrow = false }: { label: string; value: string; grow?: boolean; mobileGrow?: boolean }) {
  return <div className={grow ? 'min-w-0 flex-1' : mobileGrow ? 'min-w-0 flex-1 sm:flex-none' : ''}><span className="type-label block">{label}</span><strong className="mt-1 block max-w-[240px] truncate text-[13px] font-medium leading-tight">{value}</strong></div>;
}

function WorkPackagePage({ view, servingCapacity, loading, onOpenRun }: { view: WorkPackageView | null; servingCapacity: ServingCapacityWorkPackage | null; loading: boolean; onOpenRun: (runKey: string) => void }) {
  if (loading) return <div className="grid min-h-[420px] place-items-center text-xs text-muted">Assembling work-package evidence…</div>;
  if (!view) return <EmptyState title="Work package unavailable" body="Observatory could not assemble the package projection from its configured evidence sources." />;
  const stage = view.runs[0]?.run.stage ?? view.work_package_id.split('/')[0] ?? 'work';
  const statusCounts = view.runs.reduce<Record<string, number>>((counts, item) => {
    counts[item.run.status] = (counts[item.run.status] ?? 0) + 1;
    return counts;
  }, {});
  const outcome = Object.entries(statusCounts).map(([status, count]) => `${count} ${status}`).join(' · ') || 'No runs';
  const lineage = [...new Map(view.lineage.map(([, artifact]) => [`${artifact.direction}:${artifact.logical_name}`, artifact])).values()];
  const consumed = lineage.filter((artifact) => artifact.direction === 'input');
  const produced = lineage.filter((artifact) => artifact.direction === 'output');
  return <>
    <PageHeading eyebrow={`${stage.toUpperCase()} WORK PACKAGE`} title={packageLabel(view.work_package_id)} subtitle={view.description ?? "No work-package description was recorded. Missing job kinds mean not run, never zero."} />
    <section aria-label="Work package summary" className="obs-card mt-5 grid overflow-hidden sm:grid-cols-2 xl:grid-cols-5">
      {[
        ['Project', view.project_id ? projectLabel(view.project_id) : 'Not recorded'],
        ['Stage', stage],
        ['Runs', String(view.runs.length)],
        ['Job kinds', String(view.job_groups.length)],
        ['Run outcomes', outcome],
      ].map(([label, value]) => <div key={label} className="min-w-0 border-b border-r border-divider px-4 py-3"><span className="type-label">{label}</span><strong title={value} className="mt-1.5 block truncate text-[11px] font-medium text-ink">{value}</strong></div>)}
    </section>
    {servingCapacity && (servingCapacity.contenders?.length ?? 0) > 0 && <ServingCapacityWorkPackageView view={servingCapacity} onOpenRun={onOpenRun} />}
    <div className="mt-6 grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_340px]">
      <section aria-labelledby="package-run-groups">
        <div className="mb-3"><h2 id="package-run-groups" className="font-serif text-xl font-normal">Run groups</h2><p className="mt-1 text-xs leading-5 text-muted">Runs are grouped by reusable job kind so retries and variants stay inside the package boundary.</p></div>
        <div className="space-y-3">{view.job_groups.map((group) => {
          const groupRuns = view.runs.filter((run) => group.run_keys.includes(run.run_key));
          return <article key={group.job_kind} className="obs-card overflow-hidden">
            <header className="flex flex-wrap items-start justify-between gap-3 border-b border-divider bg-subtle/50 px-4 py-3"><div className="min-w-0 max-w-3xl"><p className="type-label">JOB KIND</p><h3 className="mt-1 font-mono text-[12px] font-medium">{group.job_kind}</h3><div className="mt-2 space-y-1.5">{group.definitions.map((definition) => <div key={definition.id} className="text-[10px] leading-4"><code className="text-violet-700">{definition.id}</code><p className="mt-0.5 text-secondary">{definition.description ?? 'No job-definition description was recorded.'}</p></div>)}</div></div><div className="flex items-center gap-2 text-[10px] text-muted"><span>{groupRuns.length} {groupRuns.length === 1 ? 'run' : 'runs'}</span>{[...new Set(group.statuses)].map((status) => <Status key={status} value={status} />)}</div></header>
            <div>{groupRuns.map((item) => <button key={item.run_key} type="button" aria-label={`Open run ${item.run.display_name}`} onClick={() => onOpenRun(item.run_key)} className="grid w-full gap-2 border-b border-divider px-4 py-3 text-left last:border-b-0 hover:bg-subtle sm:grid-cols-[minmax(0,1fr)_120px_110px_auto] sm:items-center">
              <span className="min-w-0"><strong className="block truncate text-[12px] font-medium">{item.run.display_name}</strong><code title={item.run.run_id} className="mt-1 block truncate text-[9px] text-muted">{item.run.run_id}</code></span>
              <Status value={item.run.status} />
              <span className="text-[10px] text-muted">{formatTimestamp(item.run.started_at)}</span>
              <span className="inline-flex items-center gap-1 text-[10px] text-muted">{item.metric_names.length} metrics <ArrowRight size={12} aria-hidden="true" /></span>
            </button>)}</div>
          </article>;
        })}</div>
      </section>
      <aside className="space-y-4">
        <section aria-labelledby="package-lineage" className="obs-card overflow-hidden">
          <header className="border-b border-divider px-4 py-3"><p className="type-eyebrow">PACKAGE EVIDENCE</p><h2 id="package-lineage" className="mt-1 font-serif text-xl">Artifact lineage</h2></header>
          <div className="grid grid-cols-2 border-b border-divider"><ContextValue label="Consumed" value={String(consumed.length)} /><ContextValue label="Produced" value={String(produced.length)} /></div>
          {lineage.length ? <div className="p-3">{lineage.slice(0, 8).map((artifact) => <div key={`${artifact.direction}:${artifact.logical_name}`} className="flex items-start gap-2 border-b border-divider px-1 py-2 last:border-b-0"><TreeStructure size={15} className="mt-0.5 shrink-0 text-violet-700" aria-hidden="true" /><span className="min-w-0"><strong className="block text-[10px] font-medium capitalize">{artifact.direction} · {artifactLabel(artifact.kind)}</strong><code title={artifact.logical_name} className="mt-0.5 block truncate text-[9px] text-muted">{artifact.logical_name}</code></span></div>)}</div> : <p className="p-4 text-xs leading-5 text-muted">No artifact edges were recorded for these runs.</p>}
        </section>
        <section aria-label="Decision record" className="obs-card border-dashed p-4"><p className="type-eyebrow">DECISION RECORD</p><h2 className="mt-1 font-serif text-lg">No package conclusion recorded</h2><p className="mt-2 text-[11px] leading-5 text-muted">The current evidence projection does not expose the package owner, decision question, or conclusion. Observatory keeps that absence explicit instead of inferring a decision from run success.</p></section>
      </aside>
    </div>
  </>;
}

function Overview({ selected, response, copy, activeChart, onChart, onTraces }: {
  selected: RunItem;
  response: RunView;
  copy: { eyebrow: string; title: string; question: string };
  activeChart: number;
  onChart: (index: number) => void;
  onTraces: () => void;
}) {
  const view = response.view;
  const summary = view.summary ?? [];
  const charts = view.charts ?? [];
  const helpByMetric = useMemo(
    () => new Map((view.metric_help ?? []).map((item) => [item.metric, item])),
    [view.metric_help],
  );
  const chartLabels = useMemo(
    () => Object.fromEntries((view.metric_help ?? []).map((item) => [item.metric, item.label])),
    [view.metric_help],
  );
  const chart = charts[Math.min(activeChart, Math.max(charts.length - 1, 0))];
  const lead = summary[0];
  const leadPoints = charts
    .flatMap((item) => item.series)
    .find((series) => series.name === lead?.metric)?.points ?? [];
  const previousLeadPoint = leadPoints.at(-2);
  const latestLeadPoint = leadPoints.at(-1);
  const leadDelta = previousLeadPoint && latestLeadPoint
    ? latestLeadPoint.value - previousLeadPoint.value
    : null;
  const latestStep = chart?.series[0]?.points.at(-1)?.step ?? null;
  const [selectedStep, setSelectedStep] = useState<number | null>(latestStep);
  useEffect(() => setSelectedStep(latestStep), [activeChart, latestStep, selected.run.run_id]);
  const selectedSeries = chart?.series.map((series) => ({
    name: series.name,
    value: series.points.find((point) => point.step === selectedStep)?.value ?? null,
  })) ?? [];
  const model = selectionValue(view.resolved_inputs, 'model');
  const student = selectionValue(view.resolved_inputs, 'student')
    ?? selectionValue(view.resolved_inputs, 'student_model');
  const teacher = selectionValue(view.resolved_inputs, 'teacher')
    ?? selectionValue(view.resolved_inputs, 'teacher_model');
  const inference = selectionValue(view.resolved_inputs, 'inference');
  const dataset = selectionValue(view.resolved_inputs, 'dataset');
  const validationDataset = selectionValue(view.resolved_inputs, 'validation_dataset');
  const training = selectionValue(view.resolved_inputs, 'training');
  const settings = selectionValue(view.resolved_inputs, 'settings');
  const jobDefinition = view.resolved_inputs ? configEntry(view.resolved_inputs, 'job_definition') : null;
  const jobDefinitionDescription = typeof jobDefinition?.detail.description === 'string'
    ? jobDefinition.detail.description
    : null;
  const method = methodParameters(selected.run.job_kind, dataset, validationDataset, settings, training);
  const executionTargets = view.execution_targets ?? [];
  const inputArtifacts = view.artifacts.items.filter((artifact) => artifact.direction === 'input');
  const outputArtifacts = view.artifacts.items.filter((artifact) => artifact.direction === 'output');
  const isSft = selected.run.job_kind === 'train.sft';
  const isDpo = selected.run.job_kind === 'train.dpo';
  const isGrpo = selected.run.job_kind === 'train.grpo';
  const isSampo = selected.run.job_kind === 'train.sampo';
  const isDistill = selected.run.job_kind === 'train.distill';
  const isServeSmoke = selected.run.job_kind === 'serve.smoke';
  const isDataPrepare = selected.run.job_kind === 'data.prepare';
  const completeness = view.completeness;
  const missingRequirement = completeness?.requirements.find(
    (item) => item.state === 'missing' && item.level !== 'diagnostic',
  );
  const healthAlert = view.alerts?.find(
    (item) => !item.id.startsWith('evidence-') && !item.id.startsWith('missing-'),
  );
  const primarySummary = isSft
    ? summary.filter((item) => ['validation_loss', 'token_accuracy', 'grad_norm', 'tokens_per_second', 'step_time'].includes(item.key))
    : isDpo
      ? summary.filter((item) => ['preference_accuracy', 'chosen_reward', 'rejected_reward', 'chosen_logp', 'grad_norm', 'entropy'].includes(item.key))
    : summary.slice(1, 7);
  const dataSummary = isSft
    ? summary.filter((item) => ['supervision_ratio', 'truncation_rate', 'max_length_utilization'].includes(item.key))
    : isDpo
      ? summary.filter((item) => ['preference_pairs', 'prompt_tokens', 'chosen_tokens', 'rejected_tokens', 'prompt_tokens_p95', 'chosen_tokens_p95', 'rejected_tokens_p95', 'max_length_headroom', 'chosen_longer_fraction', 'score_coverage', 'score_margin', 'max_length_utilization'].includes(item.key))
      : [];
  return (
    <>
      <div>
        <div><p className="type-eyebrow">{copy.eyebrow}</p><h1 className="type-page-title mt-1.5">{copy.title}</h1><p className="type-page-subtitle mt-2">{copy.question}</p>{jobDefinitionDescription && <p className="mt-2 max-w-3xl text-[10px] leading-4 text-muted"><code className="mr-2 text-violet-700">{jobDefinition?.id}</code>{jobDefinitionDescription}</p>}</div>
      </div>
      {response.fallback_reason && <div className="mt-5 flex items-center gap-2 border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800"><Warning size={16} weight="fill" />{response.fallback_reason}</div>}
      {healthAlert && <div className="obs-card mt-5 flex items-center gap-3 border-amber-200 bg-[#fffaf1] px-3 py-2 text-[11px]"><Warning size={16} weight="fill" className="text-amber-500" /><strong>Live health</strong><span className="text-secondary">{healthAlert.message}</span><button className="ml-auto text-violet-700">View evidence</button></div>}
      {(isDpo || isGrpo || isSampo || isDistill) && completeness && (
        <section aria-label={`${isGrpo ? 'GRPO' : isSampo ? 'SAMPO' : isDistill ? 'Distillation' : 'DPO'} evidence completeness`} className="obs-card mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 text-[11px]">
          <div className="flex items-center gap-2">
            <Circle size={9} weight="fill" className={completeness.state === 'complete' ? 'text-emerald-600' : completeness.state === 'partial' ? 'text-amber-500' : 'text-rose-600'} />
            <span className="type-label">EVIDENCE</span>
            <strong className="text-xs capitalize">{completeness.state}</strong>
          </div>
          <span className="text-muted">Required {completeness.required_available}/{completeness.required_total}</span>
          <span className="text-muted">Active conditional {completeness.conditional_available}/{completeness.conditional_active}</span>
          <span className={completeness.research_ready ? 'text-emerald-700' : 'text-amber-700'}>{completeness.research_ready ? 'Research ready' : 'Not research ready'}</span>
          {missingRequirement && (
            <span className="min-w-[240px] flex-1 text-right text-muted" title={missingRequirement.reason ?? undefined}>
              {missingRequirement.label} · {missingRequirement.missing_metrics.length} {missingRequirement.missing_metrics.length === 1 ? 'metric' : 'metrics'} missing
            </span>
          )}
        </section>
      )}
      {isGrpo && view.grpo && (
        <section aria-label="GRPO rollout population" className="obs-card mt-3 overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-divider px-4 py-2.5">
            <div><p className="type-eyebrow">ROLLOUT POPULATION</p><h2 className="mt-1 font-serif text-lg">Requested to usable evidence</h2></div>
            <div className="flex items-center gap-3 text-[10px] text-muted">
              <span>MTP {view.grpo.acceleration.mtp_selected ? 'selected' : 'off'}</span>
              <span>Quantized KV {view.grpo.acceleration.quantized_kv_cache_selected ? 'selected' : 'off'}</span>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5">
            {Object.values(view.grpo.rollout_population).map((metric) => <div key={metric.key} className="border-b border-r border-divider px-4 py-3"><MetricLabel label={metric.label} metric={metric.metric} help={metric.metric ? helpByMetric.get(metric.metric) : undefined} className="text-[10px] text-muted" /><strong className="mt-1 block font-serif text-xl font-normal">{formatValue(metric.value, metric.unit)}</strong></div>)}
          </div>
          {(view.grpo.acceleration.mtp_selected || view.grpo.acceleration.quantized_kv_cache_selected) && <div className="grid border-t border-divider sm:grid-cols-3">{[view.grpo.acceleration.speculative_acceptance, view.grpo.acceleration.accepted_speculative_length, view.grpo.acceleration.kv_cache_peak_usage].map((metric) => <div key={metric.key} className="border-r border-divider px-4 py-3"><MetricLabel label={metric.label} metric={metric.metric} help={metric.metric ? helpByMetric.get(metric.metric) : undefined} className="text-[10px] text-muted" /><strong className="mt-1 block font-serif text-lg font-normal">{formatValue(metric.value, metric.unit)}</strong>{metric.state !== 'available' && <small className="text-[10px] text-amber-700">Runtime evidence missing</small>}</div>)}</div>}
        </section>
      )}
      <div className="mt-5 grid gap-4 xl:grid-cols-[minmax(0,1fr)_292px]">
        <section className="min-w-0">
          {lead ? (
            <section className="obs-card overflow-hidden">
              <div className="grid border-b border-divider lg:grid-cols-[260px_minmax(0,1fr)]">
                <div className="px-5 py-4"><MetricLabel label={lead.label} metric={lead.metric} help={lead.metric ? helpByMetric.get(lead.metric) : undefined} className="text-xs text-secondary" /><div className="mt-1 flex items-end gap-3"><strong className="font-serif text-[52px] font-normal leading-none">{formatValue(lead.value, lead.unit)}</strong>{lead.state !== 'available' && <span className="pb-1 text-[11px] text-amber-700">{lead.state}</span>}</div>{leadDelta != null && <p className="mt-2 text-[10px] text-muted">{leadDelta >= 0 ? '+' : ''}{formatValue(leadDelta, lead.unit)} vs step {previousLeadPoint?.step ?? leadPoints.length - 1}</p>}</div>
                <div className="grid grid-cols-2 border-divider sm:grid-cols-3 lg:border-l">{primarySummary.map((metric) => <div key={metric.key} className="border-b border-l border-divider px-4 py-3 first:border-l-0 lg:first:border-l"><MetricLabel label={metric.label} metric={metric.metric} help={metric.metric ? helpByMetric.get(metric.metric) : undefined} className="text-[11px] text-muted" /><strong className="mt-1 block font-serif text-xl font-normal">{formatValue(metric.value, metric.unit)}</strong>{metric.state !== 'available' && <small className="text-[10px] text-amber-700">{metric.state}</small>}</div>)}</div>
              </div>
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-divider px-4 py-2.5">
                {charts.length > 1 ? <div className="inline-flex rounded-[5px] border border-divider bg-subtle p-0.5" role="tablist" aria-label="Training evidence chart"><>{charts.map((item, index) => <button key={item.key} type="button" role="tab" aria-selected={activeChart === index} onClick={() => onChart(index)} className={`rounded-[3px] px-3 py-1.5 text-xs ${activeChart === index ? 'bg-surface text-violet-800 shadow-sm' : 'text-muted hover:text-ink'}`}>{item.title}</button>)}</></div> : <span className="text-xs font-medium">{chart?.title}</span>}
                <span className="max-w-xl text-right text-[11px] text-muted">{chart?.question ?? 'Select a point to inspect exact evidence'}</span>
              </div>
              <div className="flex min-h-10 flex-wrap items-center gap-x-5 gap-y-2 border-b border-divider bg-subtle/45 px-4 py-2 text-[11px]">
                <span className="font-medium text-ink">Step {selectedStep ?? '—'}</span>
                {selectedSeries.map((item) => <span key={item.name} className="inline-flex items-center text-secondary"><MetricLabel label={helpByMetric.get(item.name)?.label ?? metricLabel(item.name)} metric={item.name} help={helpByMetric.get(item.name)} className="text-muted" /> <strong className="ml-1 font-medium text-ink">{formatValue(item.value, metricUnits[item.name] ?? helpByMetric.get(item.name)?.unit)}</strong></span>)}
              </div>
              {chart && <div className="px-2 pb-1 pt-2"><Suspense fallback={<ChartFallback height={330} />}><EvidenceChart series={chart.series} metricLabels={chartLabels} selectedStep={selectedStep} onPointSelect={setSelectedStep} ariaLabel={`${chart.title} metric series for ${selected.run.display_name}`} /></Suspense></div>}
            </section>
          ) : <EmptyState title="No registered job summary" body="Use Metrics for raw, bounded evidence. Observatory will not infer job semantics that are not registered." />}
          {dataSummary.length > 0 && (
            <section className="obs-card mt-3 overflow-hidden" aria-labelledby="data-profile-heading">
              <div className="flex flex-wrap items-end justify-between gap-3 border-b border-divider px-4 py-3">
                <div><p className="type-eyebrow">{isDpo ? 'RENDERED PREFERENCE POPULATION' : 'RENDERED TRAINING POPULATION'}</p><h2 id="data-profile-heading" className="mt-1 font-serif text-lg">{isDpo ? 'Pair profile' : 'Data utilization'}</h2></div>
                <p className="max-w-lg text-[11px] leading-4 text-muted">{isDpo ? 'Length balance and source-score coverage reveal shortcuts or ambiguous preference evidence.' : 'Computed after the selected renderer and max length, before optimization begins.'}</p>
              </div>
              <div className={`grid ${isDpo ? 'sm:grid-cols-2 xl:grid-cols-4' : 'sm:grid-cols-3'}`}>{dataSummary.map((metric) => <div key={metric.key} className="border-b border-r border-divider px-4 py-3 last:border-r-0"><MetricLabel label={metric.label} metric={metric.metric} help={metric.metric ? helpByMetric.get(metric.metric) : undefined} className="text-[11px] text-muted" /><strong className="mt-1 block font-serif text-xl font-normal">{formatValue(metric.value, metric.unit)}</strong>{metric.state !== 'available' && <small className="text-[10px] text-amber-700">Not recorded</small>}</div>)}</div>
            </section>
          )}
        </section>
        <aside className="obs-card self-start overflow-hidden">
          <div className="border-b border-divider px-4 py-3"><p className="type-eyebrow">RUN LINEAGE</p><h2 className="mt-1 font-serif text-xl">Inputs to outputs</h2></div>
          {isDistill ? <>
            <LineageItem icon={<Stack size={18} />} label="Student model" value={student?.id ?? 'Not recorded'} detail={student?.revision ? `revision ${student.revision.slice(0, 12)}` : undefined} />
            <LineageItem icon={<Stack size={18} />} label="Teacher model" value={teacher?.id ?? 'Not recorded'} detail={teacher?.revision ? `revision ${teacher.revision.slice(0, 12)}` : undefined} />
          </> : !isServeSmoke && !isDataPrepare && <LineageItem icon={<Stack size={18} />} label="Base model" value={model?.id ?? 'Not recorded'} detail={model?.revision ? `revision ${model.revision.slice(0, 12)}` : undefined} />}
          {isServeSmoke && <LineageItem icon={<Pulse size={18} />} label="Inference binding" value={inference?.id ?? 'Not recorded'} detail={inference?.revision ? `revision ${inference.revision.slice(0, 12)}` : undefined} />}
          {!isServeSmoke && <LineageItem icon={<Database size={18} />} label={isDataPrepare ? 'Source dataset' : 'Consumed dataset'} value={dataset?.id ?? 'Not recorded'} detail={dataset?.revision ? `revision ${dataset.revision.slice(0, 12)}` : undefined} />}
          {validationDataset && <LineageItem icon={<Database size={18} />} label="Validation dataset" value={validationDataset.id} detail={validationDataset.revision ? `revision ${validationDataset.revision.slice(0, 12)}` : undefined} />}
          {!isServeSmoke && !isDataPrepare && <LineageItem icon={<Pulse size={18} />} label="Training binding" value={training?.id ?? 'Not recorded'} detail={typeof training?.detail.parameter_update_kind === 'string' ? training.detail.parameter_update_kind.toUpperCase() : undefined} />}
          {executionTargets.map((target) => <LineageItem
            key={`${target.selection_id}:${target.revision ?? 'unknown'}`}
            icon={<Cpu size={18} />}
            label="Execution target"
            value={target.selection_id}
            detail={[
              target.device_count && target.device_class ? `${target.device_count} × ${target.device_class}` : target.device_class,
              target.aggregate_memory_bytes ? `${formatValue(target.aggregate_memory_bytes, 'bytes')} aggregate VRAM` : 'capacity not retained',
              target.roles.length ? target.roles.join(', ') : null,
            ].filter(Boolean).join(' · ')}
          />)}
          {inputArtifacts.length > 0 && <section aria-label="Input artifacts" className="border-b border-divider px-4 py-3"><p className="type-label">Input artifacts</p>{inputArtifacts.map((artifact) => <div key={artifact.logical_name} className="mt-3 flex gap-2.5"><TreeStructure size={18} className="mt-0.5 shrink-0 text-muted" /><div className="min-w-0"><strong className="block text-xs font-medium">{artifactLabel(artifact.kind)}</strong><small title={artifact.logical_name} className="block truncate text-[10px] text-muted">{artifact.logical_name} · {artifact.artifact.version}</small></div></div>)}</section>}
          {method.length > 0 && <section aria-label="Algorithm settings" className="border-b border-divider px-4 py-3"><p className="type-label">Algorithm settings</p><h3 className="mt-1 font-serif text-base">Method context</h3><dl className="mt-3 grid grid-cols-2 gap-x-3 gap-y-3">{method.map((item) => <div key={item.label} className="min-w-0"><dt className="text-[9px] uppercase tracking-[.12em] text-muted">{item.label}</dt><dd className="mt-0.5 break-words text-[11px] font-medium text-ink">{item.value}</dd></div>)}</dl></section>}
          <section aria-label="Produced evidence" className="border-t border-divider px-4 py-3"><p className="type-label">Produced evidence</p>{outputArtifacts.length ? outputArtifacts.map((artifact) => <div key={artifact.logical_name} className="mt-3 flex gap-2.5"><TreeStructure size={18} className="mt-0.5 shrink-0 text-violet-700" /><div className="min-w-0"><strong className="block text-xs font-medium">{artifactLabel(artifact.kind)}</strong><small title={artifact.logical_name} className="block truncate text-[10px] text-muted">{artifact.logical_name} · {artifact.artifact.version}</small></div></div>) : <p className="mt-2 text-xs text-muted">No produced artifacts recorded.</p>}</section>
          {view.trace_evaluation_enabled && !!view.trace_count && <button onClick={onTraces} className="mt-4 inline-flex items-center gap-1 text-xs text-violet-700">Inspect {view.trace_count} traces <ArrowSquareOut size={13} /></button>}
        </aside>
      </div>
    </>
  );
}

function LineageItem({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail?: string }) {
  return <div className="border-b border-divider px-4 py-3"><div className="flex gap-2.5 text-muted">{icon}<div className="min-w-0"><p className="type-label">{label}</p><strong className="mt-1 block break-words text-xs font-medium text-ink">{value}</strong>{detail && <small className="mt-1 block text-[10px] text-muted">{detail}</small>}</div></div></div>;
}

const MAX_SELECTED_METRICS = 12;

function GenericMetricCard({ metric, onRemove }: { metric: MetricSeries; onRemove: () => void }) {
  const latest = metric.points.at(-1);
  const previous = metric.points.at(-2);
  const delta = latest && previous ? latest.value - previous.value : null;
  const label = metricLabels[metric.name] ?? metricLabel(metric.name);
  return <section aria-label={`${metric.name} metric card`} className="obs-card min-w-0 overflow-hidden">
    <div className="flex items-start justify-between gap-3 border-b border-divider px-4 py-3">
      <div className="min-w-0">
        <h3 className="text-xs font-medium text-ink">{label}</h3>
        <code className="mt-1 block truncate text-[10px] text-muted" title={metric.name}>{metric.name}</code>
      </div>
      <button type="button" aria-label={`Remove ${metric.name}`} onClick={onRemove} className="rounded-[4px] p-1 text-muted hover:bg-subtle hover:text-ink"><X size={13} /></button>
    </div>
    <div className="flex items-end justify-between gap-3 px-4 pt-3">
      <strong className="font-serif text-3xl font-normal">{formatValue(latest?.value, metricUnits[metric.name])}</strong>
      <div className="text-right text-[10px] text-muted"><span className="block">{metric.points.length} points</span>{delta != null && <span className={delta > 0 ? 'text-emerald-700' : delta < 0 ? 'text-rose-700' : ''}>{delta > 0 ? '+' : ''}{formatValue(delta, metricUnits[metric.name])} latest change</span>}</div>
    </div>
    <div className="px-2 pb-1"><Suspense fallback={<ChartFallback height={190} />}><EvidenceChart series={[metric]} height={190} compact showLegend={false} ariaLabel={`${metric.name} metric series`} /></Suspense></div>
  </section>;
}

function GenericMetrics({ response, runKey }: { response: RunView; runKey: string }) {
  const catalog = response.view.metric_catalog;
  const names = useMemo(() => catalog?.namespaces.flatMap((namespace) => namespace.metrics) ?? [], [catalog]);
  const initial = useMemo(() => names.filter((name) => !name.startsWith('system/')).slice(0, 3), [names]);
  const [selected, setSelected] = useState<string[]>(initial);
  const [series, setSeries] = useState(response.view.selected_series ?? null);
  const [loadingSeries, setLoadingSeries] = useState(false);
  const [query, setQuery] = useState('');
  useEffect(() => {
    setSelected(initial);
    setSeries(null);
    setQuery('');
  }, [initial, runKey]);
  useEffect(() => {
    if (!selected.length) {
      setSeries(null);
      return;
    }
    let cancelled = false;
    setLoadingSeries(true);
    api.view(runKey, 'generic', selected)
      .then((value) => {
        if (!cancelled) setSeries(value.view.selected_series ?? null);
      })
      .finally(() => {
        if (!cancelled) setLoadingSeries(false);
      });
    return () => { cancelled = true; };
  }, [runKey, selected]);
  if (!catalog) return <EmptyState title="Generic view unavailable" body="Return to Overview and reopen Metrics to request the generic evidence projection." />;
  const normalizedQuery = query.trim().toLowerCase();
  const visibleNamespaces = catalog.namespaces
    .map((namespace) => ({ ...namespace, metrics: namespace.metrics.filter((metric) => metric.toLowerCase().includes(normalizedQuery)) }))
    .filter((namespace) => namespace.metrics.length > 0);
  return <><PageHeading eyebrow="RAW, BOUNDED EVIDENCE" title="Metric workspace" subtitle="Select recorded metrics for independent, unit-safe inspection. Each metric keeps its own card, scale, and latest value." /><div className="mt-6 grid gap-4 xl:grid-cols-[292px_minmax(0,1fr)]"><aside className="obs-card self-start p-4"><div className="flex items-center justify-between"><h2 className="text-[13px] font-medium">Metric catalog</h2><span className="text-[11px] text-muted">{selected.length}/{MAX_SELECTED_METRICS} selected</span></div><label className="obs-control mt-3 flex h-8 items-center gap-2 px-2.5"><MagnifyingGlass size={13} /><input aria-label="Filter metrics" value={query} onChange={(event) => setQuery(event.target.value)} placeholder={`${catalog.total} recorded metrics`} className="w-full bg-transparent text-[11px] outline-none" /></label><div className="mt-4 max-h-[640px] space-y-4 overflow-auto pr-1">{visibleNamespaces.map((namespace) => <section key={namespace.name}><h3 className="type-label">{namespace.name}</h3><div className="mt-2 space-y-0.5">{namespace.metrics.map((metric) => { const checked = selected.includes(metric); const atLimit = !checked && selected.length >= MAX_SELECTED_METRICS; return <label key={metric} className={`flex items-start gap-2 border-b border-divider py-2 text-xs ${atLimit ? 'cursor-not-allowed text-muted' : 'cursor-pointer'}`} title={atLimit ? `Remove a card before selecting more than ${MAX_SELECTED_METRICS} metrics.` : undefined}><input type="checkbox" checked={checked} disabled={atLimit} onChange={() => setSelected((values) => checked ? values.filter((name) => name !== metric) : [...values, metric])} className="mt-0.5 accent-violet-700" /><code className="break-all">{metric}</code></label>; })}</div></section>)}</div></aside><section aria-label="Selected metric cards" className="min-w-0"><div className="mb-3 flex items-end justify-between gap-3"><div><h2 className="text-[13px] font-medium">Selected metrics</h2><p className="mt-1 text-xs text-muted">Raw evidence only; no job-specific health judgment is applied.</p></div>{series && <span className="text-[11px] text-muted">{series.returned_points}/{series.requested_points} points{series.downsampled ? ' · downsampled' : ''}</span>}</div>{loadingSeries && !series ? <div className="grid gap-4 md:grid-cols-2"><div className="obs-card"><ChartFallback height={280} /></div><div className="obs-card"><ChartFallback height={280} /></div></div> : series?.series.length ? <><div className="grid gap-4 md:grid-cols-2">{series.series.map((metric) => <GenericMetricCard key={metric.name} metric={metric} onRemove={() => setSelected((values) => values.filter((name) => name !== metric.name))} />)}</div>{loadingSeries && <p className="mt-3 text-[11px] text-muted">Refreshing selected metrics…</p>}</> : <div className="obs-card"><EmptyState title="Select a metric" body={`Choose up to ${MAX_SELECTED_METRICS} recorded metrics. Each selection appears as an independent card.`} /></div>}</section></div></>;
}

function SystemView({ system }: { system: SystemMetrics | null }) {
  if (!system) return <EmptyState title="Loading system telemetry" body="Fetching canonical system/* and tracking/* evidence for this run." />;
  const heading = <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
    <PageHeading eyebrow="CROSS-JOB RUNTIME EVIDENCE" title="System metrics" subtitle={system.state === 'unavailable' ? "Host telemetry is queried only for the selected run's start-to-finish window." : "Host telemetry is restricted to this run's start-to-finish window so machine activity is not attributed across jobs."} />
    <section aria-label="Run telemetry window" className="pb-0.5"><dl className="flex flex-wrap items-center justify-end gap-x-4 gap-y-1 text-[10px] text-muted">
      <div className="flex items-baseline gap-1.5"><dt className="text-[9px] uppercase tracking-[.1em]">Started</dt><dd title={formatTimestamp(system.window_started_at)}>{system.window_started_at ? formatSidebarTimestamp(system.window_started_at) : 'In progress'}</dd></div>
      <div className="flex items-baseline gap-1.5"><dt className="text-[9px] uppercase tracking-[.1em]">Finished</dt><dd title={formatTimestamp(system.window_finished_at)}>{system.window_finished_at ? formatSidebarTimestamp(system.window_finished_at) : 'In progress'}</dd></div>
      <div className="flex items-baseline gap-1.5"><dt className="text-[9px] uppercase tracking-[.1em]">Samples</dt><dd>{system.sample_count.toLocaleString()}</dd></div>
    </dl></section>
  </div>;
  if (system.state === 'unavailable') return <>{heading}<div className="obs-card mt-5 border-amber-200 bg-[#fffaf1] p-5"><h2 className="text-[13px] font-medium">No host samples were found in this window</h2><p className="mt-2 max-w-3xl text-xs leading-5 text-secondary">The selected evidence backend has no existing host telemetry for this run interval. Observatory leaves the evidence missing instead of attributing samples from another run or synthesizing boundary values.</p></div></>;
  const helpByMetric = new Map(system.summary.map((metric) => [metric.metric, metric]));
  const labels = Object.fromEntries(system.summary.map((metric) => [metric.metric, metric.label]));
  return <>{heading}<PhaseMemoryTimeline system={system} /><div className="obs-card mt-4 grid overflow-hidden sm:grid-cols-2 xl:grid-cols-6">{system.summary.map((metric) => <div key={metric.key} className="border-b border-r border-divider px-4 py-3"><MetricLabel label={metric.label} metric={metric.metric} help={metric} className="text-[11px] text-muted" /><strong className="mt-1 block font-serif text-xl font-normal">{formatValue(metric.value, metric.unit)}</strong>{metric.state !== 'available' && <small className="text-[10px] text-amber-700">{metric.state}</small>}</div>)}</div><div className="mt-4 grid gap-4 xl:grid-cols-2">{system.groups.map((group) => <section key={group.key} aria-label={`${group.title} system chart`} className={`obs-card p-4 ${group.key === 'runtime' ? 'xl:col-span-2' : ''}`}><div className="flex flex-wrap items-center gap-x-3 gap-y-1"><h2 className="text-[13px] font-medium">{group.title}</h2><span className="hidden h-3 w-px bg-divider sm:block" aria-hidden="true" /><div className="flex flex-wrap items-center gap-x-2 text-[10px] text-muted">{group.series.map((item) => <MetricLabel key={item.name} label={helpByMetric.get(item.name)?.label ?? metricLabel(item.name)} metric={item.name} help={helpByMetric.get(item.name)} />)}</div></div><Suspense fallback={<ChartFallback />}><EvidenceChart series={group.series} metricLabels={labels} compact height={250} ariaLabel={`${group.title} over run time`} /></Suspense></section>)}</div>{system.missing.length > 0 && <p className="mt-4 text-xs text-muted">Not recorded: {system.missing.join(', ')}</p>}</>;
}

function TraceView({ jobKind, evaluation, detail, onSelect }: { jobKind: string; evaluation: TraceEvaluation | null; detail: TraceDetail | null; onSelect: (trace: TraceSummary) => Promise<void> }) {
  const [slice, setSlice] = useState('all');
  const [outcome, setOutcome] = useState('all');
  const [query, setQuery] = useState('');
  const [sorting, setSorting] = useState<SortingState>([{ id: 'reward', desc: false }]);
  const traces = useMemo(() => evaluation?.traces.filter((trace) => (slice === 'all' || trace.task === slice) && (outcome === 'all' || (outcome === 'pass' ? trace.success === true : trace.success === false)) && (!query || `${trace.external_id} ${trace.task}`.toLowerCase().includes(query.toLowerCase()))) ?? [], [evaluation, outcome, query, slice]);
  if (!evaluation) return <EmptyState title="Loading trace population" body="Scanning a bounded trace population and computing aggregate views." />;
  if (!evaluation.included) return <EmptyState title="No traces were captured" body="This job has run-level evidence only. Trace-derived evaluation and example-level investigation are unavailable." />;
  const activeFilters = Number(slice !== 'all') + Number(outcome !== 'all') + Number(Boolean(query));
  const isGrpo = jobKind === 'train.grpo';
  return <><PageHeading eyebrow={isGrpo ? 'ROLLOUTS TO LEARNING SIGNAL' : 'AGGREGATE TO EVIDENCE'} title={isGrpo ? 'Rollouts & rewards' : 'Traces & evaluation'} subtitle={isGrpo ? 'Reward and rollout aggregates link back to the exact trajectories consumed by policy updates.' : 'Evaluation metrics are computed from this trace population; every aggregate links back to exact examples.'} /><div className="obs-card mt-5 grid overflow-hidden sm:grid-cols-3 xl:grid-cols-6">{[['Mean reward', evaluation.mean_reward?.toFixed(3) ?? '—'], ['Pass rate', evaluation.success_rate == null ? '—' : `${(evaluation.success_rate * 100).toFixed(1)}%`], ['Included', evaluation.included], ['Errors', evaluation.failures], ['Truncated', evaluation.truncated], ['Trace sync', evaluation.state]].map(([label, value]) => <div key={label} className="border-b border-r border-divider px-4 py-3"><span className="block text-[11px] text-muted">{label}</span><strong className="mt-1 block font-serif text-2xl font-normal">{value}</strong></div>)}</div><div className="mt-3"><Suspense fallback={<ChartFallback height={230} />}><EvaluationCharts evaluation={evaluation} /></Suspense></div><div className="obs-card mt-3 flex flex-wrap items-center gap-2 px-2.5 py-2 text-xs"><SlidersHorizontal size={15} className="mx-0.5 text-muted" /><FilterPopover label="Slice" value={slice} onChange={setSlice} options={[{ value: 'all', label: 'Any' }, ...evaluation.slices.map((item) => ({ value: item.key, label: item.key }))]} /><FilterPopover label="Outcome" value={outcome} onChange={setOutcome} options={[{ value: 'all', label: 'Any' }, { value: 'pass', label: 'Pass' }, { value: 'review', label: 'Needs review' }]} />{activeFilters > 0 && <button type="button" onClick={() => { setSlice('all'); setOutcome('all'); setQuery(''); }} className="inline-flex h-8 items-center gap-1 px-2 text-[11px] text-violet-700 hover:text-violet-900"><X size={12} /> Clear {activeFilters}</button>}<label className="obs-control ml-auto flex h-8 min-w-[250px] items-center gap-2 px-2.5 focus-within:border-violet-400"><MagnifyingGlass size={13} /><input aria-label="Search traces" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search traces, prompts, IDs…" className="w-full bg-transparent outline-none" /></label></div><div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_390px]"><div><div className="mb-2 flex items-end justify-between"><div><h2 className="text-[13px] font-medium">Trace population</h2><p className="mt-1 text-xs text-muted">{traces.length.toLocaleString()} of {evaluation.included.toLocaleString()} traces · sorted and filtered locally.</p></div></div><Suspense fallback={<ChartFallback height={430} />}><TraceTable traces={traces} selectedId={detail?.summary.external_id ?? null} sorting={sorting} onSortingChange={setSorting} onSelect={(trace) => void onSelect(trace)} /></Suspense></div><TraceInspector detail={detail} /></div></>;
}

function TraceInspector({ detail }: { detail: TraceDetail | null }) {
  if (!detail) return <aside className="obs-card p-5"><p className="text-xs text-muted">Select a trace to inspect its transcript and verifier evidence.</p></aside>;
  return <aside className="obs-card overflow-hidden"><div className="border-b border-divider p-4"><p className="type-label">SELECTED TRACE</p><h2 className="mt-1 font-mono text-xs">{detail.summary.external_id}</h2><div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs"><Status value={detail.summary.success ? 'complete' : 'failed'} /><span>Reward {detail.summary.reward?.toFixed(3) ?? '—'}</span><span>{detail.summary.task}</span><span>{detail.summary.latency_ms == null ? '—' : `${(detail.summary.latency_ms / 1000).toFixed(1)}s`}</span><span>{detail.summary.tokens?.toLocaleString() ?? '—'} tokens</span></div></div><div className="max-h-[520px] overflow-auto p-4"><h3 className="type-label">Rollout transcript</h3><div className="mt-2 space-y-2">{detail.transcript.map((message, index) => <div key={index} className="rounded-[4px] border border-divider bg-surface p-2.5"><span className="text-[10px] font-medium uppercase tracking-[.1em] text-violet-700">{String(message.role ?? 'event')}</span><p className="mt-1 text-xs leading-5 text-secondary">{String(message.content ?? JSON.stringify(message))}</p></div>)}</div><h3 className="type-label mt-5">Reward components</h3><div className="mt-2 space-y-2">{detail.reward_components.map((item) => <div key={item.name} className="grid grid-cols-[1fr_2fr_auto] items-center gap-2 text-xs"><span>{item.name}</span><meter min="-1" max="1" value={item.value} className="w-full" /><strong>{item.value.toFixed(3)}</strong></div>)}</div><h3 className="type-label mt-5">Metadata</h3><pre className="mt-2 overflow-auto rounded-[4px] bg-subtle p-3 text-[11px] leading-4">{JSON.stringify(detail.attributes, null, 2)}</pre></div></aside>;
}

type RunInputItem = {
  label: string;
  value: ConfigEntryValue;
  icon: ReactNode;
};

function ArtifactsView({ response }: { response: RunView }) {
  const artifacts = response.view.artifacts.items;
  const inputs = artifacts.filter((artifact) => artifact.direction === 'input');
  const outputs = artifacts.filter((artifact) => artifact.direction === 'output');
  const run = response.view.run;
  const inputSnapshot = response.view.resolved_inputs ?? {};
  const executionTargets = response.view.execution_targets ?? [];
  const resolvedInputs: RunInputItem[] = [
    { label: 'Base model', value: configEntry(inputSnapshot, 'model'), icon: <Stack size={17} /> },
    { label: 'Training dataset', value: configEntry(inputSnapshot, 'dataset'), icon: <Database size={17} /> },
    { label: 'Validation dataset', value: configEntry(inputSnapshot, 'validation_dataset'), icon: <Database size={17} /> },
  ].flatMap((item) => item.value ? [{ ...item, value: item.value }] : []);
  resolvedInputs.push(...executionTargets.map((target) => ({
    label: 'Execution target',
    icon: <Cpu size={17} />,
    value: {
      id: target.selection_id,
      revision: target.revision,
      sourceLayer: null,
      overlayId: null,
      detail: {
        device_class: target.device_class,
        device_count: target.device_count,
        aggregate_memory_bytes: target.aggregate_memory_bytes,
        roles: target.roles,
      },
    },
  })));
  return <>
    <PageHeading eyebrow="RUN INPUTS & ARTIFACT EDGES" title="Artifacts & lineage" subtitle="Review resolved model and data inputs beside recorded immutable artifact edges and durable outputs." />
    <section aria-labelledby="lineage-heading" className="obs-card mt-6 overflow-hidden">
      <div className="flex flex-wrap items-end justify-between gap-3 border-b border-divider px-5 py-4">
        <div>
          <h2 id="lineage-heading" className="text-[13px] font-medium">Run inputs and outputs</h2>
          <p className="mt-1 text-xs text-muted">{resolvedInputs.length} resolved input{resolvedInputs.length === 1 ? '' : 's'} · {inputs.length} recorded input edge{inputs.length === 1 ? '' : 's'} · {outputs.length} recorded output edge{outputs.length === 1 ? '' : 's'}</p>
        </div>
        <span className="type-label">resolved inputs → run → output artifacts</span>
      </div>
      <div className="grid items-stretch gap-3 p-4 xl:grid-cols-[minmax(0,1fr)_32px_minmax(220px,.8fr)_32px_minmax(0,1.2fr)]">
        <RunInputLane inputs={resolvedInputs} artifacts={inputs} />
        <LineageArrow />
        <article className="flex min-h-[150px] flex-col justify-between rounded-[5px] border border-violet-300 bg-violet-50 p-4" aria-label={`Run ${run.display_name}`}>
          <div>
            <p className="type-label text-violet-700">Observed run</p>
            <h3 className="mt-2 font-serif text-xl font-normal leading-tight">{run.display_name}</h3>
            <p className="mt-2 text-xs text-secondary">{run.job_kind} · {run.job_definition_version}</p>
          </div>
          <div className="mt-5 flex items-center justify-between gap-3 border-t border-violet-200 pt-3">
            <Status value={run.status} />
            <code title={run.run_id} className="truncate text-[10px] text-violet-700">{run.run_id.slice(0, 12)}</code>
          </div>
        </article>
        <LineageArrow />
        <LineageLane title="Produced artifacts" direction="output" artifacts={outputs} empty={run.status === 'failed' ? 'The failed run did not publish durable output artifacts.' : 'No produced artifact edges were recorded for this run.'} />
      </div>
    </section>
    <section aria-labelledby="artifact-ledger-heading" className="mt-6">
      <div className="flex items-end justify-between gap-4">
        <div><h2 id="artifact-ledger-heading" className="text-[13px] font-medium">Artifact ledger</h2><p className="mt-1 text-xs text-muted">Provider identity, immutable version, and digest for every recorded edge.</p></div>
        <span className="text-[11px] text-muted">{artifacts.length} artifact{artifacts.length === 1 ? '' : 's'}</span>
      </div>
      {artifacts.length ? <div className="obs-card mt-3 overflow-hidden">{artifacts.map((artifact) => <ArtifactLedgerRow key={`${artifact.direction}:${artifact.logical_name}`} artifact={artifact} />)}</div> : <div className="obs-card mt-3 p-5 text-xs text-muted">This run has no artifact edges. The absence is preserved as missing lineage rather than inferred from configuration.</div>}
    </section>
  </>;
}

function RunInputLane({ inputs, artifacts }: {
  inputs: RunInputItem[];
  artifacts: Artifact[];
}) {
  return <section aria-label="Run inputs" className="min-w-0">
    <div className="mb-2 flex items-center justify-between"><h3 className="type-label">Run inputs</h3><span className="rounded-full bg-subtle px-2 py-0.5 text-[10px] text-muted">{inputs.length}</span></div>
    {inputs.length ? <div className="space-y-2">{inputs.map((input) => <article key={input.label} className="rounded-[5px] border border-divider bg-surface p-3">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 shrink-0 text-violet-700" aria-hidden="true">{input.icon}</span>
        <div className="min-w-0">
          <p className="type-label">{input.label}</p>
          <code title={input.value.id} className="mt-1 block truncate text-[10px] text-secondary">{input.value.id}</code>
          <p className="mt-2 text-[9px] text-muted">{input.value.revision ? `revision ${input.value.revision.slice(0, 12)}` : 'revision not recorded'}{input.value.sourceLayer ? ` · ${input.value.sourceLayer}` : ''}</p>
        </div>
      </div>
    </article>)}</div> : <div className="grid min-h-[116px] place-items-center rounded-[5px] border border-dashed border-divider bg-subtle/60 p-4 text-center text-xs leading-5 text-muted">No resolved model or dataset inputs were recorded for this run.</div>}
    <div className="mt-3 border-t border-divider pt-3">
      <div className="mb-2 flex items-center justify-between"><h4 className="type-label">Recorded input artifacts</h4><span className="text-[10px] text-muted">{artifacts.length}</span></div>
      {artifacts.length ? <div className="space-y-2">{artifacts.map((artifact) => <LineageArtifact key={`input:${artifact.logical_name}`} artifact={artifact} />)}</div> : <p className="text-[10px] leading-4 text-muted">No immutable input artifact edges were recorded.</p>}
    </div>
  </section>;
}

function LineageArrow() {
  return <div className="grid place-items-center text-violet-500" aria-hidden="true"><ArrowRight size={18} className="rotate-90 xl:rotate-0" /></div>;
}

function LineageLane({ title, direction, artifacts, empty }: { title: string; direction: Artifact['direction']; artifacts: Artifact[]; empty: string }) {
  return <section aria-label={title} className="min-w-0">
    <div className="mb-2 flex items-center justify-between"><h3 className="type-label">{title}</h3><span className="rounded-full bg-subtle px-2 py-0.5 text-[10px] text-muted">{artifacts.length}</span></div>
    {artifacts.length ? <div className="space-y-2">{artifacts.map((artifact) => <LineageArtifact key={`${direction}:${artifact.logical_name}`} artifact={artifact} />)}</div> : <div className="grid min-h-[116px] place-items-center rounded-[5px] border border-dashed border-divider bg-subtle/60 p-4 text-center text-xs leading-5 text-muted">{empty}</div>}
  </section>;
}

function LineageArtifact({ artifact }: { artifact: Artifact }) {
  const metadata = artifact.artifact.provider_metadata ?? {};
  const originKind = typeof metadata.job_kind === 'string' ? metadata.job_kind : null;
  const originRun = typeof metadata.run_id === 'string' ? metadata.run_id : null;
  return <article className="rounded-[5px] border border-divider bg-surface p-3">
    <div className="flex items-start gap-2.5">
      <TreeStructure size={17} className="mt-0.5 shrink-0 text-violet-700" aria-hidden="true" />
      <div className="min-w-0">
        <p className="text-[11px] font-medium text-ink">{artifactLabel(artifact.kind)}</p>
        <code title={artifact.logical_name} className="mt-1 block truncate text-[10px] text-secondary">{artifact.logical_name}</code>
        <p className="mt-2 text-[10px] text-muted">{artifact.artifact.provider} · {artifact.artifact.version}{artifact.artifact.digest ? ` · ${artifact.artifact.digest.slice(0, 8)}` : ''}</p>
        {artifact.direction === 'input' && (originKind || originRun) && <p className="mt-2 border-t border-divider pt-2 text-[10px] text-muted">Produced by {originKind ?? 'recorded run'}{originRun ? ` · ${originRun.slice(0, 8)}` : ''}</p>}
      </div>
    </div>
  </article>;
}

function ArtifactLedgerRow({ artifact }: { artifact: Artifact }) {
  const reference = `${artifact.artifact.provider}/${artifact.artifact.namespace}/${artifact.artifact.name}:${artifact.artifact.version}`;
  return <article className="grid gap-3 border-b border-divider px-4 py-3 last:border-b-0 md:grid-cols-[110px_minmax(0,1fr)_minmax(260px,.9fr)] md:items-center">
    <span className={`w-fit rounded-full px-2 py-1 text-[10px] font-medium uppercase tracking-[.08em] ${artifact.direction === 'input' ? 'bg-sky-50 text-sky-700' : 'bg-violet-50 text-violet-700'}`}>{artifact.direction}</span>
    <div className="min-w-0"><strong className="block text-xs font-medium">{artifactLabel(artifact.kind)}</strong><code title={artifact.logical_name} className="mt-1 block truncate text-[10px] text-muted">{artifact.logical_name}</code></div>
    <div className="min-w-0 text-[10px] text-muted"><code title={reference} className="block truncate">{reference}</code><span className="mt-1 block">digest {artifact.artifact.digest ?? 'not recorded'}</span></div>
  </article>;
}

function ConfigValue({ value, depth = 0 }: { value: unknown; depth?: number }) {
  if (value == null) return <span className="text-muted">Not set</span>;
  if (typeof value === 'boolean') return <span>{value ? 'Enabled' : 'Disabled'}</span>;
  if (typeof value === 'number') {
    const formatted = value !== 0 && Math.abs(value) < 0.001
      ? value.toExponential(2)
      : value.toLocaleString(undefined, { maximumFractionDigits: 8 });
    return <span className="font-mono text-[10px]">{formatted}</span>;
  }
  if (typeof value === 'string') return <code title={value} className="break-words font-mono text-[10px] text-secondary">{value}</code>;
  if (Array.isArray(value)) {
    return <div className="space-y-1">
      <span className="text-[10px] text-muted">{value.length.toLocaleString()} {value.length === 1 ? 'item' : 'items'}</span>
      {value.length > 0 && <div className="flex flex-wrap gap-1">{value.slice(0, 3).map((item, index) => <span key={`${String(item)}:${index}`} className="max-w-full truncate rounded bg-subtle px-1.5 py-0.5 font-mono text-[9px] text-secondary">{isRecord(item) || Array.isArray(item) ? 'Structured item' : String(item)}</span>)}{value.length > 3 && <span className="rounded bg-subtle px-1.5 py-0.5 text-[9px] text-muted">+{value.length - 3} more</span>}</div>}
    </div>;
  }
  if (isRecord(value)) {
    if (depth >= 2) return <span className="text-[10px] text-muted">{Object.keys(value).length} recorded fields</span>;
    return <ConfigFields value={value} depth={depth + 1} />;
  }
  return <span>{String(value)}</span>;
}

function ConfigFields({ value, depth = 0 }: { value: Record<string, unknown>; depth?: number }) {
  const entries = Object.entries(value).filter(([, field]) => field != null);
  if (!entries.length) return <p className="bg-surface px-3 py-2.5 text-[10px] text-muted">No resolved fields were recorded.</p>;
  const visible = entries.slice(0, 14);
  return <div className={depth > 0 ? 'grid gap-px overflow-hidden rounded-[4px] border border-divider bg-divider sm:grid-cols-2' : 'grid gap-px bg-divider sm:grid-cols-2'}>
    {visible.map(([key, field]) => {
      const structured = isRecord(field);
      return <div key={key} className={`${structured ? 'sm:col-span-2' : ''} min-w-0 bg-surface px-3 py-2.5`}>
        <dt className="type-label">{humanizeKey(key)}</dt>
        <dd className="mt-1 min-w-0 text-[11px] leading-4 text-ink"><ConfigValue value={field} depth={depth} /></dd>
      </div>;
    })}
    {entries.length > visible.length && <p className="bg-surface px-3 py-2.5 text-[10px] text-muted sm:col-span-2">{entries.length - visible.length} additional fields are available in the redacted JSON.</p>}
  </div>;
}

function ConfigSelectionCard({ name, value }: { name: string; value: ConfigEntryValue }) {
  const label = selectionLabels[name] ?? humanizeKey(name);
  const wide = name === 'training' || name === 'work_package';
  return <article aria-label={`${label} selection`} className={`obs-card min-w-0 overflow-hidden ${wide ? 'lg:col-span-2' : ''}`}>
    <header className="border-b border-divider bg-subtle/50 px-4 py-3">
      <div className="flex min-w-0 items-start justify-between gap-3">
        <div className="min-w-0"><p className="type-label">{label}</p><code title={value.id} className="mt-1.5 block truncate font-mono text-[11px] font-medium text-ink">{value.id}</code></div>
        <Stack size={17} className="shrink-0 text-violet-700" aria-hidden="true" />
      </div>
      {(value.revision || value.sourceLayer || value.overlayId) && <div className="mt-2 flex flex-wrap gap-1.5 text-[9px] text-muted">
        {value.revision && <span title={value.revision} className="rounded border border-divider bg-surface px-1.5 py-0.5">revision {value.revision.length > 16 ? `${value.revision.slice(0, 12)}…` : value.revision}</span>}
        {value.sourceLayer && <span className="rounded border border-divider bg-surface px-1.5 py-0.5">{value.sourceLayer}</span>}
        {value.overlayId && <span title={value.overlayId} className="rounded border border-divider bg-surface px-1.5 py-0.5">overlay {value.overlayId}</span>}
      </div>}
    </header>
    <dl className="bg-divider"><ConfigFields value={value.detail} /></dl>
  </article>;
}

function ConfigGroup({ definition, inputs }: { definition: ConfigGroupDefinition; inputs: Record<string, unknown> }) {
  const entries = definition.keys.flatMap((key) => {
    const value = configEntry(inputs, key);
    return value ? [{ key, value }] : [];
  });
  if (!entries.length) return null;
  return <section aria-labelledby={`config-group-${definition.title.replaceAll(' ', '-').toLowerCase()}`}>
    <div className="mb-3 max-w-3xl">
      <h2 id={`config-group-${definition.title.replaceAll(' ', '-').toLowerCase()}`} className="font-serif text-xl font-normal text-ink">{definition.title}</h2>
      <p className="mt-1 text-xs leading-5 text-muted">{definition.description}</p>
    </div>
    <div className="grid items-start gap-3 lg:grid-cols-2">{entries.map((entry) => <ConfigSelectionCard key={entry.key} name={entry.key} value={entry.value} />)}</div>
  </section>;
}

function ConfigView({ response }: { response: RunView }) {
  const inputs = response.view.resolved_inputs ?? {};
  const source = response.view.source_metadata ?? {};
  const run = response.view.run;
  const groups = configGroups(run.job_kind, inputs);
  const raw = { selections: inputs, source_metadata: source };
  return <>
    <PageHeading eyebrow="REPRODUCIBLE INPUTS" title="Run configuration" subtitle="Resolved selections are organized by this job’s schema. Values remain server-redacted and provider-neutral." />
    <section aria-label="Run contract" className="obs-card mt-5 grid overflow-hidden sm:grid-cols-2 xl:grid-cols-4">
      {[
        ['Job kind', run.job_kind],
        ['Job definition', run.job_definition_version],
        ['Stage', run.stage],
        ['View schema', `v${response.view.schema_version ?? 1}`],
      ].map(([label, value]) => <div key={label} className="min-w-0 border-b border-r border-divider px-4 py-3"><span className="type-label">{label}</span><code title={value} className="mt-1.5 block truncate font-mono text-[10px] text-ink">{value}</code></div>)}
    </section>
    {Object.keys(inputs).length ? <div className="mt-7 space-y-8">{groups.map((group) => <ConfigGroup key={group.title} definition={group} inputs={inputs} />)}</div> : <div className="mt-6"><EmptyState title="No resolved selections" body="This run did not expose a redacted configuration projection. Its identity and recorded source provenance remain available below." /></div>}
    {Object.keys(source).length > 0 && <section aria-labelledby="source-provenance" className="mt-8">
      <div className="mb-3"><h2 id="source-provenance" className="font-serif text-xl font-normal text-ink">Source provenance</h2><p className="mt-1 text-xs leading-5 text-muted">The code revision and working-tree state recorded when the run started.</p></div>
      <div className="obs-card overflow-hidden"><dl><ConfigFields value={source} /></dl></div>
    </section>}
    <details className="obs-card mt-8 overflow-hidden">
      <summary className="cursor-pointer select-none px-4 py-3 text-[11px] font-medium text-secondary hover:bg-subtle">View redacted JSON</summary>
      <pre className="max-h-[520px] overflow-auto border-t border-divider bg-subtle p-4 font-mono text-[10px] leading-5 text-secondary">{JSON.stringify(raw, null, 2)}</pre>
    </details>
  </>;
}

function PageHeading({ eyebrow, title, subtitle }: { eyebrow: string; title: string; subtitle: string }) {
  return <div><p className="type-eyebrow">{eyebrow}</p><h1 className="type-page-title mt-1.5">{title}</h1><p className="type-page-subtitle mt-2">{subtitle}</p></div>;
}
