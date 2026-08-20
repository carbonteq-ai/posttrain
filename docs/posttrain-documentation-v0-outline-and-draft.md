# **Posttrain Documentation**

*v0 — outline and documentation draft*

This document proposes the structure of the Posttrain documentation and drafts the pages that define the primary user journey: understand the framework, install it, run a first job, author a work package, fine-tune a model, evaluate the result, and inspect the evidence.

The intended reader already understands basic model fine-tuning, but has not used Posttrain. The draft therefore explains Posttrain-specific vocabulary before relying on it, keeps the runnable path close to the top of each page, and moves exhaustive field-by-field detail into future reference pages.

| Note: The outline describes the eventual documentation site. The draft that follows is deliberately selective: it fully develops the highest-value onboarding and workflow pages while leaving lower-frequency reference pages at outline depth. |
| :---- |

**Where this lives.** These pages belong in the Posttrain repository, next to the code they describe, so a change to a command, schema, or default ships in the same pull request as its documentation. This document proposes the structure. The next step is to move it into the repository and edit it there.

# **1\. Documentation outline**

The documentation should be organized around the way a user actually encounters Posttrain: first understand the product, then get one job to succeed, then learn the reusable workflow, and only then move into specialized training methods, qualification, observability, and operator reference.

## **1.1 Proposed navigation**

| Start Here  ├── What is Posttrain?  ├── Mental Model  ├── Installation  └── QuickstartGuides  ├── Working with Posttrain  │     ├── Work Packages  │     ├── The Catalog  │     └── Configuration  ├── Data  │     ├── Datasets  │     ├── Environments  │     └── Workloads  ├── Training  │     ├── Training Overview  │     ├── Supervised Fine-Tuning  │     ├── Preference Optimization  │     ├── Online RL  │     ├── On-Policy Distillation  │     └── Checkpoints and Model Handoff  ├── Evaluation  │     ├── Evaluation Overview  │     ├── Evaluation Plans  │     └── Reading Results  ├── Serving and Model Transformation  │     ├── Serving Smoke Tests  │     ├── Capacity Benchmarks  │     └── Quantization  └── Execution        ├── Running Jobs Locally        ├── Running Jobs on dstack        ├── Managing Runs        └── The ControllerObservability  ├── Observatory  ├── Reading a Run  ├── Metrics  ├── Traces, Artifacts, and Lineage  └── TrackioHow-to Guides  ├── Train, Then Evaluate  ├── Compare Two Models  ├── Resume an Interrupted Run  ├── Clean Up and Delete Runs  ├── Fix Trust and Network Failures  └── TroubleshootingReference  ├── CLI  ├── Job Kinds  ├── Catalog Schema  ├── Work Package Schema  ├── Configuration Files  ├── Metrics Reference  ├── Compatibility and Support Boundaries  └── Glossary |
| :---- |

## **1.2 What each section is for**

| Section | Reader outcome | Representative pages |
| :---- | :---- | :---- |
| Start Here | Understand Posttrain and complete a first successful job. | Overview, concepts, installation, quickstart |
| Guides | Perform the main post-training workflows without needing to understand internals first. | Work packages, data, SFT, DPO/RL, evaluation, execution |
| Observability | Read a run as evidence: learning, system behavior, configuration, outputs, and provenance. | Observatory, Reading a Run, metrics, lineage |
| How-to | Solve one concrete operator task from beginning to end. | Train→evaluate, compare models, resume, cleanup, trust/network |
| Reference | Look up exact contracts when authoring or debugging. | CLI, schemas, job kinds, metric names, support boundaries |

## **1.3 Page priorities**

| Priority | Pages | Why they belong in the first documentation release |
| :---- | :---- | :---- |
| P0 | What is Posttrain? · Mental Model · Installation · Quickstart | Without these, a new user cannot build the right mental model or reach first success. |
| P0 | Work Packages · Catalog · Configuration | These are the reusable abstractions behind every job family. |
| P0 | Training Overview · SFT · Checkpoints | SFT is the most direct training path and introduces the artifact handoff model. |
| P0 | Evaluation Overview · Plans · Reading Results | Qualification must be separated from training metrics so model decisions are evidence-based. |
| P0 | Observatory · Reading a Run | Users need a practical way to interpret learning, system, config, and lineage evidence. |
| P1 | DPO · Online RL · Distillation · Serving · Quantization · Execution | Important breadth, but easier to learn once the common workflow is established. |
| P1 | High-frequency how-to guides | Turn the mental model into repeatable end-to-end operating procedures. |
| P2 | Full Reference | Necessary for completeness, but should not dominate the beginner path. |

# **2\. Documentation draft**

The pages below are written as they could appear in the documentation site. They focus on the first complete user journey and the highest-value concepts.

## **2.1 What is Posttrain?**

Posttrain is a framework for reproducible model post-training. It takes a base model through **screening, training, and qualification**, and keeps the configuration, inputs, outputs, metrics, and lineage needed to explain how each result was produced.

It is not a new training algorithm. Posttrain coordinates the tools that already do the heavy lifting—trainers, rollout engines, evaluation environments, tracking, and GPU execution—behind one stable job workflow. The value is the process around the training: deciding what will run, freezing the inputs, recording what happened, and making later comparisons defensible.

### **What Posttrain can do**

Posttrain covers the main operational stages around modern model post-training. It can prepare supervised or preference data; run supervised fine-tuning, DPO, GRPO/DAPO, multi-turn SAMPO, and on-policy distillation; execute serving smoke tests and capacity benchmarks; evaluate local models, managed deployments, or compatible remote endpoints through Verifiers environments; transform models with supported quantization paths; run work locally or on remote GPU infrastructure; retain checkpoints and model artifacts; and expose metrics, traces, system telemetry, configuration, and lineage through its evidence stack.

Those capabilities are connected by the same project, catalog, work-package, run, and artifact model. A user does not need one ad-hoc script for training, another convention for evaluation, another spreadsheet for experiment identity, and a separate memory of which checkpoint fed which benchmark. The same versioned bindings and run identity follow the work from input selection to produced evidence.

### **Why this matters to ML teams**

The industry value is less about inventing another optimizer and more about removing process failure around expensive ML experimentation. Training and evaluation stacks often combine fast-changing libraries, private datasets, chat templates, inference settings, GPU environments, and internal services. A result can become difficult to reproduce as soon as one of those inputs moves. Comparisons can also look valid while quietly changing a dataset revision, target, renderer, dependency commit, or evaluation criterion.

Posttrain makes those inputs explicit before the GPU work starts, freezes the dependency and job context, and records evidence and artifact lineage afterward. That reduces wasted accelerator time from late configuration failures, makes model comparisons easier to defend, makes handoffs between training and evaluation explicit, and gives reviewers a common record for answering what ran, how it behaved, what it produced, and whether the result is actually comparable to another run.

### **The three stages**

| Stage | Question | Typical work |
| :---- | :---- | :---- |
| Screen | Is this model and runtime worth pursuing? | Capability checks, data validation, serving smoke tests, capacity checks |
| Train | Can we produce a better model variant? | SFT, preference optimization, online RL, distillation, quantization |
| Qualify | Is the resulting variant ready to use? | General/domain evaluation, performance checks, evidence review |

The distinction matters because each stage asks a different question. A training run can be healthy without producing a better model. A qualification run is where you decide whether the new variant actually meets the task criteria.

### **What Posttrain coordinates**

| Tool | Role |
| :---- | :---- |
| Posttrain | Job identity, reproducible inputs, packaging, evidence, artifact lineage, lifecycle coordination |
| TRL / veRL | Trainer implementations for supported post-training methods |
| vLLM | Inference and rollout generation |
| Verifiers | Executable evaluation/RL environments and native traces |
| Trackio | Recording run evidence |
| Observatory | Reading, aggregating, comparing, and presenting retained evidence |
| dstack | Remote GPU scheduling and placement |

| Note: Trackio is the recording layer. Observatory is the human-facing evidence layer. Neither one produces the training objective or evaluation score by itself. |
| :---- |

### **Backend forks and release pinning**

Posttrain sits on top of several fast-moving ML backends, so a reproducible release cannot depend only on broad package version ranges. The release records backend and environment identities in a fork ledger and pins the relevant repositories to exact commits in the release constraints file.

The current constraints examples explicitly include the CarbonTeq TRL fork at \`github.com/carbonteq-ai/trl\`, Verifiers pinned to an exact \`PrimeIntellect-ai/verifiers\` commit, and CarbonTeq's \`verifiers-environments\` repository for packaged environments such as GSM8K. The release ledger can classify dependencies by how they enter the system \- direct package, runtime image kind, source overlay, vendored environment package, or deployed service. Not every pinned repository should be described as a CarbonTeq fork; the exact repository and commit in the release constraints are the source of truth.

This pinning matters because trainer or environment behavior can change without your work-package YAML changing. Exact commits make it possible to answer which backend code produced a result and to rebuild the same dependency closure later. Runtime images are likewise verified by immutable identity rather than by a moving tag.

### **Important boundaries**

* The shipped SFT path is text-only; it does not provide a vision-language training pipeline.  
* Online RL is synchronous in the current release.  
* Jobs do not automatically feed newly trained weights into later jobs. A produced model must be pinned into the catalog before a later work package can bind it.  
* Linux is the qualified execution baseline. Other platforms may be useful for authoring or CPU-only checks, but not as the primary execution target.

| Tip: If you only remember one design idea: Posttrain tries to make the identity of a run explicit before expensive work starts, and to make its outputs traceable after it finishes. |
| :---- |

## **2.2 Mental model**

Posttrain uses terms that do not map onto ordinary training scripts. In a normal script the inputs are implicit: the model is a string, the dataset is a path, the dependency versions are whatever resolved that morning. Posttrain names each of them as a versioned selection you make on purpose. The vocabulary follows from that.

Read this page before the Quickstart.

### **The building blocks**

| Term | What it is |
| :---- | :---- |
| Project | Your repository, created by \`posttrain init\`. Owns a \`.posttrain/\` directory. |
| Catalog | A versioned store of every input you can choose from: models, datasets, targets, and six other families. |
| Selection | One entry in the catalog, addressed by id and revision. |
| Seat | A named slot that a job needs filled, such as \`model\` or \`dataset\`. |
| Binding | Your choice of which selection goes into a seat. |
| Recipe | The ordered list of jobs to run, and the seats they require. |
| Work package | One question you want answered, with its recipe and bindings. The main thing you author. |
| Job | One executable step inside a work package, such as \`train.sft\`. |
| Run | One execution of a job. Resolved inputs go in; evidence comes out. |
| Artifact | Immutable output with metadata: a model, dataset, evaluation bundle, or benchmark result. |
| Evidence | The record of a run: metrics, traces, errors, and which artifacts it used and produced. |
| Lineage | The graph those artifact links form. It answers "where did this model come from?" |

### **Hierarchy**

| Project  └── Work package        "Does SFT improve task behavior?"        ├── Recipe        which jobs run and which seats they need        ├── Bindings      which catalog selections fill those seats        └── Job → Run                  ├── metrics and traces                  └── consumed / produced artifacts |
| :---- |

A project holds many work packages. A work package asks one question and lists its jobs. Executing a job produces a run. A run produces evidence and artifacts.

### **References, not copies**

A binding stores an id. It does not copy the selection's values.

| bindings:  target:    type: ref    family: target    id: targets/local-cuda-8gb |
| :---- |

The device class, memory size, and placement settings are not in this file. They live in the catalog entry that \`targets/local-cuda-8gb\` names.

Two work packages that bind \`targets/local-cuda-8gb\` use the same target, and Posttrain resolves the id to confirm it. Had both files copied the values instead, matching them would mean reading both and trusting that neither had drifted.

| Tip: This is why the catalog exists. It makes "these two runs used the same input" a fact the system verifies rather than a claim the author asserts. |
| :---- |

### **Jobs and runs**

A job is a step in a recipe. It is declared, not executed. A run is one execution of a job, with its own id, timestamps, and evidence.

Retrying a failed job produces a second run of the same job. The job does not change.

A job carries two identifiers:

* **Kind** is the category of operation, such as \`train.sft\` or \`eval.general\`.  
* **Definition** is the exact versioned implementation, such as \`train/trl-sft@1\`.

### **Plan, pack, run**

Every job passes through three steps. \`work-package run\` performs all three.

| Step | Question it answers | What happens |
| :---- | :---- | :---- |
| Plan | What would actually run? | Every binding resolves to a concrete, versioned input. No provider is contacted and no GPU is requested. |
| Pack | Can this exact plan be frozen? | The resolved inputs, source, wheels, and locks are built into a content-addressed image. |
| Run | Where does it execute? | The packed image is submitted locally or to remote GPU infrastructure. |

If an input changed after planning, packing fails. It does not build a different job under the same identity. Work packages covers the three steps in practice.

### **Artifacts, evidence, and lineage**

A run produces three kinds of record.

* **Artifacts** are immutable outputs: a trained adapter, a recovery checkpoint, an evaluation bundle, a benchmark result. Each has its own identity and version.  
* **Evidence** is what was observed during execution: metrics over time, traces, errors, truncations, system telemetry, and the resolved configuration.  
* **Lineage** is the graph of artifacts a run consumed and produced.

Lineage answers which run produced an adapter, which dataset and base model fed that run, and whether a quantized model derives from a specific checkpoint.

| Note: A produced model is not automatically available to later jobs. Pin the artifact into the catalog as a new selection and bind it in a new work package. See How-to: train, then evaluate. |
| :---- |

### **Project layout**

| Location | Contains | Commit it? |
| :---- | :---- | :---- |
| .posttrain/project.toml | Project identity, paths, execution defaults | Yes |
| .posttrain/catalog/ | Project catalog overlays | Yes |
| .posttrain/work\_packages/ | Work packages | Yes |
| .posttrain/state/ | Caches, scratch state, recovery data | No |
| \~/.config/posttrain/ | Machine settings and credentials | No |

Committed files carry your decisions. Machine settings, caches, and credentials stay on the machine. A colleague who clones the repository gets the project, not your infrastructure.

## **2.3 Installation**

Install Posttrain on the machine you use to author or execute jobs. The normal path uses the internal package index; the release wheelhouse is the recovery/offline path.

### **Requirements**

| Requirement | Why you need it |
| :---- | :---- |
| Python 3.13 | Supported interpreter for the current release |
| uv | Environment and package management |
| Docker with Buildx | Packing and local job execution |
| Linux | Qualified execution baseline |
| NVIDIA GPU | Only required when training locally |
| gh | Only required for the release-wheelhouse download path |

### **Trust the internal CA first**

Internal package, registry, and tracking services use a private certificate authority. Configure host trust before installing; otherwise the first failure you see is usually TLS rather than anything Posttrain-specific.

| sudo cp /path/to/internal-ca.crt   /usr/local/share/ca-certificates/internal-ca.crtsudo update-ca-certificatessudo install \-D \-m 644 /path/to/internal-ca.crt   /etc/posttrain/trust/internal-ca.pem |
| :---- |

| Warning: The job-container trust file should contain the internal authority by itself. Do not merge an entire system CA bundle into it. Three trust planes Certificate setup has three separate trust boundaries. Plane 1 is the host: browsers, curl, \`uv\`, Python, and the Docker daemon read the machine CA store. Plane 2 is Docker/BuildKit while it pulls base images and pushes packed job images. Plane 3 is the running job container when it contacts internal services such as the registry or tracking endpoint. A successful check in one plane does not prove the others. The host CA installation above establishes Plane 1\. The \`/etc/posttrain/trust/internal-ca.pem\` file is the authority Posttrain passes into job images for Plane 3\. BuildKit/registry trust must also be valid for packing and pushing to succeed. The reliable end-to-end proof is a job that starts, writes evidence to the tracking service, and reconciles cleanly. Keep the framework trust file to the internal authority alone. The job image merges that authority with the certificates already present in the image; copying the host's entire trust bundle would make job behavior depend on which machine happened to pack it. |
| :---- |

### **Install from the internal index**

| uv venv \--python 3.13 .venvVIRTUAL\_ENV=.venv uv pip install \--system-certs   \--index-url https://pypi.lan/carbonteq/stable/+simple/   \--constraint github-constraints.txt   "posttrain\[observatory,trackio,trl\]" |
| :---- |

Add the \`dstack\` extra if this machine submits remote GPU jobs.

| Note: The constraints file is part of the release contract. Some maintained dependencies are pinned to exact Git commits; installing without the matching constraints file can fail resolution or create an environment different from the one the release was qualified against. |
| :---- |

### **Verify the install**

| posttrain versionposttrain doctor |
| :---- |

A healthy setup should report \`OK\` for the checks that apply to your machine. If registry-related checks show a warning, first confirm that the expected environment/configuration has been loaded rather than treating the warning as a framework failure.

## **2.4 Quickstart: your first successful job**

The goal of this page is simple: get from an installed CLI to one completed run with retained evidence. You can learn the deeper model later.

### **1\. Configure the machine**

| posttrain machine init   \--trackio-endpoint https://trackio.lan   \--python-index-url https://pypi.lan/carbonteq/stable/+simple/   \--job-registry registry.lan/carbonteq |
| :---- |

Machine configuration is shared across projects. Credentials are stored separately from the endpoint configuration.

| printf '%s\\n' 'TRACKIO\_WRITE\_TOKEN=\<token\>'   \> \~/.config/posttrain/credentials/trackio.envchmod 600 \~/.config/posttrain/credentials/\*.env |
| :---- |

### **2\. Create a project**

| posttrain init my-project \--template sftposttrain machine project add "$PWD/my-project"cd my-project |
| :---- |

The \`sft\` template gives you a runnable starting package. A \`grpo\` template is also available for online RL.

### **3\. Check readiness**

| posttrain doctorposttrain runtime images verify |
| :---- |

### **4\. Inspect what will run**

Validation and planning are intentionally cheap. They do not request a GPU and are the right place to catch missing bindings or unexpected inputs.

| posttrain catalog listposttrain work-package validate .posttrain/work\_packages/sft.yamlposttrain work-package plan .posttrain/work\_packages/sft.yaml \--job train |
| :---- |

| Tip: Use \`plan\` whenever you find yourself asking 'what exactly is this package going to run?' It resolves the inputs before the expensive part begins. |
| :---- |

### **5\. Materialize data and run**

| posttrain dataset materialize datasets/posttrain-sft-smoke@1posttrain work-package run .posttrain/work\_packages/sft.yaml \--job train |
| :---- |

| Warning: Remote training requires an explicit wall-clock timeout. Without it, planning is rejected before submission rather than allowing an unbounded remote job. |
| :---- |

### **6\. Follow and reconcile**

| posttrain run status \--lastposttrain run logs \--last \--followposttrain run reconcile \--last |
| :---- |

Reconciliation joins provider state with retained evidence. A healthy terminal result reports a consistent reconciliation and no missing required artifact roles.

### **7\. Read the evidence**

| posttrain observatory up |
| :---- |

Open the run you just produced. Start with the Overview, then use Metrics, System metrics, Artifacts & lineage, and Run config when you need more detail.

| Note: Do not treat the quickstart as the complete operating model. Its purpose is first success. The next pages explain why the package is structured the way it is and how to author your own. |
| :---- |

## **2.5 Work packages**

A work package is the unit of work you author. It should answer **one question**: can this model train under these conditions, does this variant meet a qualification plan, is this serving configuration viable, and so on.

### **A minimal work package**

| \# .posttrain/work\_packages/cpu\_check.yamlproject\_id: my-model-projectwork\_package\_id: screen/cpu-checkstage: screendescription: Validate the local CPU execution target.recipe:  type: inline  id: recipes/cpu-check@1  revision: "1"  stage: screen  seats:    target: target  jobs:    \- id: validate      kind: data.prepare      definition: data/cpu-check@1bindings:  target:    type: ref    family: target    id: targets/local-cpuenabled\_optional\_jobs: \[\]metadata:  question: Can this project resolve and execute its local configuration? |
| :---- |

### **Read the file from top to bottom**

* \`stage\` is one of \`screen\`, \`train\`, or \`qualify\`.  
* \`recipe.seats\` declares the named inputs the recipe expects and which catalog family can fill each one.  
* \`recipe.jobs\` lists the executable steps, with an id, job kind, and definition.  
* \`bindings\` fills each seat using a versioned catalog reference.  
* \`metadata.question\` states what you are trying to learn from the work package.

| Tip: If the question naturally contains two independent decisions, you probably have two work packages. The package should stay small enough that its evidence answers one thing cleanly. |
| :---- |

| Warning: Outcomes do not belong in package metadata. The file describes what you intend to run; conclusions belong in the evidence produced after execution. |
| :---- |

### **Validate and run**

| posttrain work-package validate .posttrain/work\_packages/cpu\_check.yamlposttrain work-package run .posttrain/work\_packages/cpu\_check.yaml \--job validate |
| :---- |

### **Running the steps individually**

\`work-package run\` plans, packs, and submits in one command. Each step can also be invoked on its own.

| posttrain job plan .posttrain/work\_packages/cpu\_check.yaml \--job validateposttrain job pack .posttrain/work\_packages/cpu\_check.yaml \--job validateposttrain job run  .posttrain/work\_packages/cpu\_check.yaml \--job validate |
| :---- |

Use \`plan\` while authoring. It resolves every binding and reports what would execute. No provider is contacted and no GPU is requested, so it is free to repeat.

\`posttrain job diff\` compares two packed jobs and reports why their identities differ.

| Warning: If an input changes after planning, packing fails instead of silently building a different job under the same identity. That failure is intentional: discovering a mismatch before a multi-hour GPU run is cheaper than explaining it afterward. |
| :---- |

### **Jobs in one package do not chain**

Two jobs in the same recipe share the same bindings. An evaluation job does **not** receive the weights produced by a training job earlier in the recipe.

To use a newly trained model in a later package:

1. Run the training job and let it publish the model artifact.  
2. Read the artifact's immutable identity.  
3. Register that artifact as a model selection in your catalog overlay.  
4. Bind the new model selection in a separate work package.

| Warning: Bind a concrete immutable artifact version such as \`v3\`, not a moving alias such as \`latest\`. |
| :---- |

## **2.6 The Catalog**

The catalog is where every selectable input to a run gets a stable identity. Instead of copying model paths, target settings, or evaluation definitions into each package, you define them once and bind them by reference.

| posttrain catalog listposttrain catalog show model models/qwen3.5-0.8b@bf16posttrain catalog validate |
| :---- |

### **Layers and overlays**

The framework provides a base catalog layer. Your project adds an overlay that introduces project-specific selections or overrides individual entries.

| \# .posttrain/catalog/project/layer.yamlschema\_version: 1layer\_id: my-model-project-v1files:  \- models.yaml  \- targets.yaml |
| :---- |

| Warning: A catalog file that exists on disk but is not listed in \`layer.yaml\` is invisible to Posttrain. This is one of the easiest authoring mistakes to make. |
| :---- |

### **The nine selection families**

| Family | What it describes |
| :---- | :---- |
| Model | Concrete weights: hub snapshot, adapter, merged model, or quantized model |
| Dataset | Supervised or preference data and its source/materialization contract |
| Target | Where the job runs: device class, memory, placement |
| Training | Optimization settings and training binding |
| Quantization | Transform method, calibration, output form |
| Inference | Engine, renderer, and generation settings |
| Workload | Serving benchmark prompt population |
| Environment | Versioned Verifiers package and activation configuration |
| Evaluation | A plan that composes environments with success criteria |

A recipe declares **seats**; a work package fills those seats with catalog selections. That is the bridge from reusable definitions to a concrete run.

## **2.7 Training overview**

Training methods differ in their data and objective, but the user-facing workflow should stay consistent: bind the model and method inputs, validate the package, run it, watch method-appropriate evidence, and hand the produced model forward through the catalog.

| Method | Job kind | Use it when |
| :---- | :---- | :---- |
| Supervised fine-tuning | \`train.sft\` | You have examples of the target response behavior. |
| Preference optimization | \`train.dpo\` | You have chosen/rejected response pairs and want the model to prefer one behavior. |
| Online RL | \`train.grpo\` | The model can generate live rollouts and an environment can score them. DAPO is selected within this job kind. |
| Multi-turn online RL | \`train.sampo\` | The learning signal depends on multi-turn trajectories. |
| On-policy distillation | \`train.distill\` | A teacher can provide token-level guidance to the student policy. |

Every method binds a settings selection containing a shared training loop. Fields such as maximum steps, sequence length, device batch size, gradient accumulation, learning rate, warmup, checkpoint cadence, and seed are validated before execution.

### **Update plans**

| Plan | What changes | Typical use |
| :---- | :---- | :---- |
| Full | All model weights | When full fine-tuning is intended and memory permits |
| LoRA | A small adapter | Default adapter-style fine-tuning path |
| QLoRA | LoRA adapter over a quantized base | When memory pressure makes full-precision base weights expensive |
| Quantization-aware | Intended fake-quantization path | Not implemented by the current TRL adapter |

## **2.8 Supervised Fine-Tuning**

Use SFT when you already have examples of the behavior you want the model to reproduce. Posttrain's SFT path is conversation-oriented: it renders the example using the model family's chat format, then trains only on the messages you marked as supervised targets.

### **Quick start**

| posttrain work-package run .posttrain/work\_packages/sft.yaml \--job train |
| :---- |

The SFT template binds the base model, the supervised dataset, training-loop settings, the training/backend binding, and optionally a held-out validation dataset.

| Seat | Supplies |
| :---- | :---- |
| model | The base weights to update |
| dataset | Supervised conversations |
| settings | Loop settings such as steps, length, batch size, learning rate |
| training | Backend, renderer, update plan, and target |
| validation\_dataset | Optional held-out data |

### **How an example becomes training input**

| dataset example  messages \+ trainable\_message\_indices \+ tools        │        ▼renderer  model-family chat format        │        ▼input\_ids \+ labels  labels \= \-100 outside trainable target messages        │        ▼TRL SFTTrainer |
| :---- |

Messages outside the supervised target still provide context, but their labels are masked to \`-100\`, so they do not contribute loss. This is how you can train on the assistant response while preserving the system/user context needed to interpret it.

| Warning: If rendering leaves an example with no trainable tokens, the run errors instead of silently dropping the row. Fix the target-message indices or increase the available sequence length. |
| :---- |

### **Truncation is evidence, not a hidden preprocessing detail**

Examples longer than \`max\_length\` are truncated. Posttrain records the original and retained lengths so you can see whether your supervision is being cut rather than discovering it indirectly from model behavior.

### **LoRA and QLoRA artifacts**

Adapter training deliberately produces different artifacts for two different jobs:

| Artifact | Contains | Use it for |
| :---- | :---- | :---- |
| Model adapter | Adapter weights that can be loaded with the referenced base model | Evaluation, serving, downstream model use |
| Training checkpoint | Adapter plus optimizer, scheduler, and RNG state | Exact resume after interruption |

| Note: A recovery checkpoint is not automatically a model variant. The loadable model artifact is what enters normal model lineage. |
| :---- |

### **What to watch while SFT runs**

Open the run in Observatory and answer four questions rather than staring at one loss curve.

| Question | Signals | What a problem can look like |
| :---- | :---- | :---- |
| Is it learning? | Training loss · validation loss · token accuracy | Training loss falls while held-out loss flattens or rises |
| Are updates stable? | Gradient norm | Sharp spikes that suggest an unstable learning rate or update |
| Is supervision intact? | Supervision-token ratio · truncated examples | Target tokens are being cut or most tokens are context rather than supervised output |
| Is throughput holding? | Non-padding tokens/s · step time | Throughput falls without an intentional configuration change |

Data utilization is computed on the rendered training population after the renderer and \`max\_length\`, but before optimization:

* **Supervision-token ratio** — the fraction of rendered tokens that actually carry loss.  
* **Truncated-example rate** — the fraction of examples cut by the sequence-length limit.  
* **Max-length utilization** — how much of the available context window the rendered examples consume.

| Tip: A low supervision-token ratio is not automatically wrong. Long context with a short answer naturally produces one. The useful question is whether that ratio matches the task you intended to train. |
| :---- |

### **SFT boundary: text-only training**

| Warning: The shipped \`train.sft\` path is token-based end to end. Catalog metadata may describe image-capable architectures and some checkpoints may require multimodal-capable loaders, but the training pipeline does not pass image tensors to the trainer. Multimodal/VLM SFT is therefore not supported by this release. |
| :---- |

### **Where the other training pages go next**

The full documentation should give DPO, online RL, and distillation pages the same shape as SFT. Their key differences are:

| Page | What must be explained |
| :---- | :---- |
| Preference Optimization | Chosen/rejected data; reward margin; why both chosen and rejected rewards matter |
| Online RL | GRPO/DAPO/SAMPO; rollout budgets; reward variance; scoreability/truncation; active sampling; clip/TIS diagnostics; synchronous boundary |
| On-Policy Distillation | Teacher/student configuration; token-level scoring; objective boundaries; student/teacher evidence |
| Checkpoints & Handoff | Recovery vs model artifacts; verification; resume; immutable catalog pinning |

## **2.9 Evaluation overview**

Training metrics tell you whether optimization is behaving. They do **not** tell you whether the model solves the problem you care about. Posttrain treats evaluation as a separate qualification workflow so the success criteria, test population, and model identity are frozen and reviewable.

### **Quick start**

| posttrain work-package run .posttrain/work\_packages/eval.yaml \--job evaluate |
| :---- |

| Job kind | Use it for |
| :---- | :---- |
| \`eval.general\` | General capability across composed Verifiers environments |
| \`eval.domain\` | Project-specific or domain-specific tasks |

An evaluation run records the population, declared success predicate, reward/metric source, task facets, sampling, and aggregation rules. That prevents a later catalog or UI change from silently redefining what an earlier result meant.

### **How evaluation uses Verifiers**

Verifiers environments are the executable task layer used by Posttrain for both qualification and online RL. An environment packages the task population, reward or metric logic, tool behavior, timeouts, and other activation settings as a versioned dependency. Posttrain pins the environment source to a repository commit and subdirectory, packages it into the job image, and records that identity with the run.

During evaluation, Posttrain runs the selected Verifiers environments against the bound model or endpoint. The environment produces its native output \- including traces, rewards or metrics, configuration, logs, and errors. Completed trace rows are ingested as they appear, while the native bundle is retained as an artifact. Selected numeric fields are then projected into bounded \`eval/\*\` metrics for dashboards and queries, and Observatory reads the same retained evidence for aggregate views and trace inspection.

The native Verifiers output remains authoritative because an environment can know details that a generic tracking schema cannot: its reward decomposition, task structure, tool semantics, and failure modes. Keeping the native trace bundle means a reviewer can go back from a pass rate or reward to the actual messages, tool calls, outputs, timing, truncations, and errors that produced it. Missing terminal evidence is reported as missing rather than converted into an invented score.

The same environment abstraction can also be used for online RL. In that case the current policy generates rollouts, Verifiers scores them, and the training backend updates on the resulting signal. This keeps the executable task and reward definition versioned across training and qualification instead of re-implementing them in two separate systems.

## **2.10 Evaluation plans**

An evaluation plan composes one or more environments and declares what counts as success for each environment.

| evaluation:  general-capability-balanced-v1:    revision: "1"    kind: general    environments:      \- knowledge-mmlu-pro-cot-5shot-balanced-v1      \- instruction-ifeval-full-v1      \- reasoning-gym-balanced-eval-v1    success:      knowledge-mmlu-pro-cot-5shot-balanced-v1:        id: answer-correct        label: Answer correct        source: {namespace: reward, name: answer\_correct}        predicate: {operator: eq, value: 1}      reasoning-gym-balanced-eval-v1:        id: full-credit-solution        label: Full-credit solution        source: {namespace: metric, name: native\_score}        predicate: {operator: gte, value: 0.99} |
| :---- |

Success is declared per environment because environments do not all express correctness the same way. A binary answer-correct reward and a continuous reasoning score should not be forced through one global threshold.

| Note: The success predicate is versioned with the plan and recorded in the run. Changing the threshold creates a new plan revision; it does not retroactively reinterpret earlier runs. |
| :---- |

## **2.11 Reading evaluation results**

The easiest way to misread an evaluation is to compress the run into one number. Keep the following dimensions separate:

| Dimension | Question |
| :---- | :---- |
| Reward / metric | What signal did the environment produce? |
| Configured success | Did that signal satisfy the declared predicate? |
| Errors | Did execution fail? |
| Truncations | Did generation hit a limit? |
| Missing signals | Was there no valid signal to score? |
| Coverage | How much of the expected population has produced evidence? |

### **Coverage is not a score**

If 40 of 200 expected traces have arrived, the run is **20% covered**. That does not mean the model is 20% correct. Coverage describes how complete the evidence is; pass rate describes the valid scored population.

| Warning: Always read coverage next to pass rate, especially while an evaluation is still running or syncing. Missing, stale, or incomparable states are not converted to zero—zero is a real measured value. |
| :---- |

### **When to open traces**

Metrics summarize. Traces explain. Open the native traces when you need to know why a rollout failed, whether reward is being gamed, whether tool calls were structurally correct, or whether the output would actually be acceptable to a user.

## **2.12 How-to: train, then evaluate**

This is the most important end-to-end workflow because it makes the model handoff explicit. Training and qualification are separate work packages with separate questions and evidence.

| Warning: Putting training and evaluation jobs in one recipe does not make the evaluation consume newly trained weights. Both jobs share the bindings that existed before execution. |
| :---- |

### **1\. Run the training package**

| posttrain work-package validate .posttrain/work\_packages/train\_sft.yamlposttrain dataset materialize datasets/my-sft-data@1posttrain work-package run .posttrain/work\_packages/train\_sft.yaml \--job trainposttrain run wait \--lastposttrain run reconcile \--last |
| :---- |

### **2\. Identify and verify the produced model**

| posttrain run show \--lastposttrain run checkpoint list \<run-id\>posttrain run checkpoint verify \<run-id\> \<step\> |
| :---- |

Record the immutable artifact identity: project, artifact name, and concrete version.

### **3\. Pin the model into the catalog**

| \# .posttrain/catalog/project/models.yamlmodel:  models/my-sft-v1:    artifact:      kind: trackio      project: \<project\>      name: \<artifact-name\>      version: v3    capabilities:      modalities: \[text\]    provenance:      parent: models/qwen3.5-0.8b@bf16 |
| :---- |

List the file in your project catalog layer, validate the catalog, and confirm that the new model selection resolves.

| posttrain catalog validateposttrain catalog show model models/my-sft-v1 |
| :---- |

| Warning: Use the concrete artifact version, never \`latest\`, when the model becomes an input to a new run. |
| :---- |

### **4\. Bind the pinned model in a qualification package**

| bindings:  model:    type: ref    family: model    id: models/my-sft-v1  evaluation\_plan:    type: ref    family: evaluation    id: general-capability-balanced-v1  target:    type: ref    family: target    id: targets/remote-a100 |
| :---- |

### **5\. Run and read**

| posttrain work-package validate .posttrain/work\_packages/qualify\_eval.yamlposttrain work-package run .posttrain/work\_packages/qualify\_eval.yaml \--job evaluateposttrain run wait \--lastposttrain run reconcile \--lastposttrain observatory up |
| :---- |

The training package answered 'can we produce a better variant?' The qualification package answers 'does this variant meet the declared criteria?' Keeping those questions separate is what makes both results interpretable.

## **2.13 Observatory**

Observatory is the read-only evidence product for Posttrain. Use it to inspect a run while it is executing or after it has finished, compare candidates, and trace an output model back to the inputs and artifacts that produced it.

### **Why Observatory is useful**

Every job run through Posttrain leaves a retained evidence trail, and Observatory turns that trail into a readable view. Runs are readable **while they execute**, so you do not have to wait for an expensive job to finish before checking whether it is behaving as intended.

The practical shift is that an experiment becomes inspectable as a complete run rather than as a scroll of terminal output. Questions that would otherwise be reconstructed from memory become things you look up:

* Is the model learning, and is held-out loss moving with training loss or away from it?  
* Did the learning-rate schedule follow the intended path?  
* Are optimization updates stable?  
* Did throughput or memory pressure change during the run?  
* What configuration, model revision, and dataset revision produced this curve?  
* Which artifacts did this run consume and produce?

For RL jobs, rollout- and reward-level views add the behavior of the policy and the signals used to update it.

| posttrain observatory up |
| :---- |

Observatory follows the same hierarchy you author: **project → work package → run**. Inside a run, the available views answer different questions rather than presenting one undifferentiated dashboard.

| View | Question it answers |
| :---- | :---- |
| Overview | Is this run healthy, judged against what this job kind is for? |
| Metrics | What exactly was recorded, without a job-specific health judgment? |
| System metrics | Where did time and memory go? |
| Rollouts & rewards | For RL jobs: what did the policy produce, and how was it scored? |
| Artifacts & lineage | What went in and what came out? |
| Run config | What exactly would I have to repeat? |

The split between **Overview** and **Metrics** is deliberate. Overview interprets a selected set of evidence for the job kind. Metrics is the raw workspace. When you need an exact recorded value or a series the Overview does not show, use Metrics.

### **Overview: read the run as a question**

For SFT, the Overview is organized around a practical question: is held-out loss improving without unstable updates, damaged supervision, or falling token throughput?

* Headline metrics surface training loss, validation loss, token accuracy, gradient norm, throughput, and step time.  
* Learning, stability, and efficiency lenses group metrics by the decision they support rather than overlaying everything on one chart.  
* Data-utilization values show supervision-token ratio, truncated-example rate, and max-length utilization.  
* The input-to-output lineage panel keeps the base model, datasets, training binding, and execution target beside the training evidence.

### **Metrics: inspect raw evidence safely**

The Metrics view is a searchable metric workspace. Selected metrics appear on independent cards with their own scales. This avoids putting a token count in the tens of millions on the same axis as a ratio between 0 and 1\.

| Tip: Use the Overview to decide where to look; use Metrics to verify the exact value that supports the decision. |
| :---- |

### **System metrics: separate capacity from quality**

System metrics break execution into phases and show time, observed activity, and memory pressure against declared device capacity. Online-RL runs can expose phases such as model loading, runtime initialization, rollout generation, actor update, and artifact export.

Runtime telemetry is diagnostic evidence. High GPU utilization or a good speculative-decoding acceptance rate can explain speed; neither one is evidence that the model became better.

| Warning: If the view labels phase coverage as partial, derived averages are based on incomplete telemetry. Treat them as partial rather than as full-run summaries. |
| :---- |

### **Artifacts & lineage: answer 'where did this come from?'**

The lineage view reads from left to right: resolved inputs → observed run → produced artifacts. It distinguishes catalog selections from artifacts actually consumed by the run, then records the immutable versions and digests of outputs.

This is where you verify that a model, adapter, checkpoint, evaluation bundle, or transformed artifact is actually derived from the run you think it is, rather than relying on naming conventions or memory.

### **Run config: answer 'what would I repeat?'**

Run configuration exposes the resolved model revision, environment/source revision, job definition, stage, execution target, and activation/configuration fields. This is the reproducibility view: it tells you what inputs must be held constant if you want to reproduce or legitimately compare the run.

## **2.14 Trackio**

Trackio is the recording backend used by Posttrain to persist run evidence. It is intentionally a shorter documentation page than Observatory because most users do not need to learn a second UI to interpret a run.

* Document how the endpoint and write credentials are configured.  
* Explain how a Posttrain run maps to the stored tracking run and artifacts.  
* Document the tracking-health metrics used to diagnose dropped traces or artifact-upload failures.  
* Explain backend switching where supported.  
* Send human inspection and comparison workflows to Observatory rather than duplicating those guides.

| Note: The conceptual boundary should stay explicit: backends produce measurements; Trackio records them; Observatory reads and presents retained evidence. |
| :---- |

Posttrain Documentation — v0 Outline and Draft