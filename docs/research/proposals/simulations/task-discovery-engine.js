/* Synthetic curriculum model. No trained weights or measured LLM results. */
(function (root) {
  "use strict";
  const CLASSES = ["Arithmetic", "Algebra", "Geometry", "Probability"];
  const PRESETS = {
    normal: "Steady learning", abilities: "Different starting abilities",
    signal: "Learning follows signal", varied: "Different learning speeds",
    hidden: "Mostly solved class",
    transfer: "Transfer unlocks geometry", forgetting: "Arithmetic regresses",
    noise: "Variance without learning", solved: "Everything is solved",
  };
  const normalize = a => {
    const sum = a.reduce((s, x) => s + x, 0);
    return sum > 0 ? a.map(x => x / sum) : a.map(() => 1 / a.length);
  };
  const mean = a => a.length ? a.reduce((s, x) => s + x, 0) / a.length : 0;
  const copy = x => JSON.parse(JSON.stringify(x));
  function random(seed) {
    let x = seed >>> 0;
    return () => {
      x += 0x6D2B79F5;
      let t = Math.imul(x ^ x >>> 15, 1 | x);
      t ^= t + Math.imul(t ^ t >>> 7, 61 | t);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function draw(items, weights, rng) {
    const probabilities = normalize(weights);
    let r = rng();
    for (let i = 0; i < items.length; i++) {
      r -= probabilities[i];
      if (r <= 0 || i === items.length - 1) return items[i];
    }
  }
  function inventory() {
    return CLASSES.flatMap((name, c) => Array.from({length: 16}, (_, j) => ({
      id: "ABGP"[c] + String(j + 1).padStart(2, "0"), c, label: name,
      seen: false, last: -1, history: [], discovery: null,
    })));
  }
  function initialModel(preset) {
    return inventory().map((t, i) => {
      const j = i % 16;
      let p = [0.48, 0.27, 0.13, 0.35][t.c] + ((j * 7) % 9) * 0.035;
      let rate = [0.28, 0.16, 0.09, 0.13][t.c] * [0.65, 1, 1.3][j % 3];
      if (["normal", "abilities", "signal", "varied"].includes(preset)) {
        p = (preset === "normal" ? [0.60, 0.35, 0.10, 0.45][t.c] : 0.15 + t.c * 0.15) +
          ((j * 7) % 9) * 0.035;
        rate = preset === "varied" ? [0.07, 0.14, 0.28][j % 3] : 0.14;
      }
      if (preset === "hidden" && t.c === 0) {
        p = j < 12 ? 0.995 : 0.25 + (j - 12) * 0.1;
        rate = j < 12 ? 0 : 0.1;
      }
      if (preset === "transfer" && t.c === 2) { p = 0.015; rate = 0.06; }
      if (preset === "noise" && t.c === 3) { p = 0.5; rate = 0; }
      if (preset === "solved") { p = 1; rate = 0; }
      return {p, rate, gain: 0};
    });
  }
  function groupSignal(outcomes) {
    const success = mean(outcomes);
    return 4 * success * (1 - success);
  }
  function directGain(p, rate, outcomes, preset) {
    const signal = groupSignal(outcomes);
    return signal === 0 ? 0 : rate * (0.995 - p) * (preset === "signal" ? signal : 1);
  }
  function recent(t, step) {
    return t.history.filter(h => step - h.step <= 8).slice(-4);
  }
  const useful = h => h.success > 0 && h.success < 4 ? 1 : 0;
  function recencyWeights(count) {
    if (!count) return [];
    const denominator = count * (count + 1) / 2;
    return Array.from({length: count}, (_, i) => (i + 1) * count / denominator);
  }
  function classPosterior(tasks, c, step, omit = "") {
    let alpha = 1, beta = 1;
    for (const t of tasks.filter(t => t.c === c && t.id !== omit)) {
      if (!t.discovery) continue;
      alpha += useful(t.discovery);
      beta += 1 - useful(t.discovery);
    }
    const total = alpha + beta;
    const mean = alpha / total;
    const uncertainty = Math.sqrt(alpha * beta / (total * total * (total + 1)));
    return {alpha, beta, mean, uncertainty, priority: Math.min(1, mean + uncertainty)};
  }
  function classSuccessMean(tasks, c, omit = "") {
    const discoveries = tasks.filter(t => t.c === c && t.id !== omit && t.discovery);
    return (1 + discoveries.reduce((sum, t) => sum + t.discovery.success / 4, 0)) /
      (2 + discoveries.length);
  }
  function mixedGroupProbability(alpha, beta, size = 4) {
    let allSuccess = 1, allFailure = 1;
    for (let offset = 0; offset < size; offset++) {
      allSuccess *= (alpha + offset) / (alpha + beta + offset);
      allFailure *= (beta + offset) / (alpha + beta + offset);
    }
    return Math.max(0, Math.min(1, 1 - allSuccess - allFailure));
  }
  function predictions(tasks, step) {
    const classes = CLASSES.map((_, c) => classPosterior(tasks, c, step));
    return {
      classes: classes.map(value => value.priority),
      classMeans: classes.map(value => value.mean),
      classUncertainty: classes.map(value => value.uncertainty),
      tasks: tasks.map(t => {
        const records = recent(t, step);
        const yieldPrior = classPosterior(tasks, t.c, step, t.id).mean;
        const successPrior = classSuccessMean(tasks, t.c, t.id);
        let successAlpha = 2 * successPrior, successBeta = 2 * (1 - successPrior);
        let yieldAlpha = 2 * yieldPrior, yieldBeta = 2 * (1 - yieldPrior);
        const weights = recencyWeights(records.length);
        records.forEach((h, i) => {
          successAlpha += weights[i] * h.success;
          successBeta += weights[i] * (4 - h.success);
          yieldAlpha += weights[i] * useful(h);
          yieldBeta += weights[i] * (1 - useful(h));
        });
        return mixedGroupProbability(successAlpha, successBeta) *
          yieldAlpha / (yieldAlpha + yieldBeta);
      }),
    };
  }
  const quota = (n, b, percent) =>
    Math.floor((n + b) * percent / 100) - Math.floor(n * percent / 100);

  function choose(tasks, scores, excluded, discovery, covered, rng) {
    // Intentionally accepts no hidden model probabilities.
    const unseen = tasks.map((t, i) => i).filter(i => !excluded.has(tasks[i].id) && !tasks[i].seen);
    const familiar = tasks.map((t, i) => i).filter(i => !excluded.has(tasks[i].id) && tasks[i].seen);
    let pool, reason, poolChance = 1;
    if (discovery && unseen.length) {
      pool = unseen; reason = "Discovery";
    } else if (unseen.length && familiar.length) {
      const unseenClasses = [...new Set(unseen.map(i => tasks[i].c))];
      const unseenScore = mean(unseenClasses.map(c => scores.classes[c]));
      const familiarScore = mean(familiar.map(i => scores.tasks[i]));
      const discoveryChance = unseenScore / (unseenScore + familiarScore);
      if (rng() < discoveryChance) {
        pool = unseen; reason = "Additional new task"; poolChance = discoveryChance;
      } else {
        pool = familiar; reason = "Adaptive practice"; poolChance = 1 - discoveryChance;
      }
    } else if (familiar.length) {
      pool = familiar; reason = "Adaptive practice";
    } else {
      pool = unseen;
      reason = discovery ? "Discovery" : "Additional new task";
    }
    let fallback = "";
    if (!pool.length) {
      pool = tasks.map((t, i) => i).filter(i => !excluded.has(tasks[i].id));
      fallback = discovery ? "inventory exhausted" : "startup / familiar pool empty";
    }
    if (!pool.length) return null;
    const unseenPool = pool.every(i => !tasks[i].seen);
    const classes = [...new Set(pool.map(i => tasks[i].c))];
    const cWeights = classes.map(c => unseenPool ? scores.classes[c] :
      mean(pool.filter(i => tasks[i].c === c).map(i => scores.tasks[i])));
    const adaptiveClasses = normalize(cWeights);
    const c = draw(classes, covered ? classes.map(() => 1) : adaptiveClasses, rng);
    const ci = classes.indexOf(c);
    const classChance = covered ? 1 / classes.length : adaptiveClasses[ci];
    const choices = pool.filter(i => tasks[i].c === c);
    const oldest = Math.min(...choices.map(i => tasks[i].last));
    const oldestTasks = choices.filter(i => tasks[i].last === oldest);
    const weights = choices.map(i => scores.tasks[i]);
    const noSignal = weights.every(w => w === 0);
    const reassess = !unseenPool && (covered || discovery || noSignal);
    const task = unseenPool ? draw(choices, choices.map(() => 1), rng) :
      reassess ? draw(oldestTasks, oldestTasks.map(() => 1), rng) : draw(choices, weights, rng);
    const withinCoverage = oldestTasks.includes(task) ? 1 / oldestTasks.length : 0;
    const withinAdaptive = noSignal ? withinCoverage : normalize(weights)[choices.indexOf(task)];
    const joint = unseenPool ? classChance / choices.length :
      classChance * (reassess ? withinCoverage : withinAdaptive);
    return {index: task, id: tasks[task].id, c, covered, classChance, chance: poolChance * joint,
      newTask: !tasks[task].seen, reserved: discovery, fallback,
      reason: unseenPool ? reason : reassess ? "Reassessment" : "Adaptive practice"};
  }

  function simulate(options = {}) {
    const config = {preset: "normal", mode: "grpo", seed: 17, steps: 30,
      coverage: 20, discovery: 20, ...options};
    const rng = random(config.seed), outcomesRng = random(config.seed ^ 0x9E3779B9);
    const tasks = inventory(), model = initialModel(config.preset);
    const frames = [], totals = {candidates: 0, newTasks: 0, reserved: 0, fulfilled: 0,
      coverageReserved: 0, coverageFulfilled: 0, reassessed: 0, attempts: 0, optimized: 0, updates: 0};
    let step = 1, round = 1, selected = [], history = [], retained = [], excluded = new Set();
    let due = 0, coverageDue = 0, roundStart = 0, shock = false;
    function snapshot(phase, message, extra = {}) {
      const scores = predictions(tasks, step);
      const classStats = CLASSES.map((name, c) => {
        const indices = tasks.map((t, i) => i).filter(i => tasks[i].c === c);
        const unseen = indices.filter(i => !tasks[i].seen).length;
        const observed = indices.flatMap(i => recent(tasks[i], step));
        return {name, unseen, prediction: scores.classMeans[c],
          uncertainty: scores.classUncertainty[c], priority: scores.classes[c],
          success: observed.length ? mean(observed.map(h => h.success / 4)) : null,
          trueMean: mean(indices.map(i => model[i].p)),
          discoveries: indices.filter(i => tasks[i].discovery && step - tasks[i].discovery.step <= 8).length};
      });
      const availableClasses = classStats.filter(c => c.unseen);
      const normal = normalize(availableClasses.map(c => c.priority));
      classStats.forEach(c => {
        const i = availableClasses.indexOf(c);
        c.discoveryShare = i < 0 ? 0 : config.coverage / 100 / availableClasses.length +
          (1 - config.coverage / 100) * normal[i];
      });
      frames.push(copy({phase, message, step, round, config, totals, due, coverageDue, roundStart,
        picks: selected, retained: retained.length, used: [...excluded], history,
        classes: classStats, shock, tasks: tasks.map((t, i) => {
          const records = recent(t, step), avg = records.length ? mean(records.map(h => h.success / 4)) : null;
          const variation = records.length ? mean(records.map(h => 4 * (h.success / 4) * (1 - h.success / 4))) : null;
          return {id: t.id, c: t.c, seen: t.seen, last: t.last, groups: t.history.length,
            recent: records, success: avg, variance: variation, predicted: scores.tasks[i],
            truth: model[i].p, lastGain: model[i].gain, state: !t.seen ? "unseen" : !records.length ? "uncertain" :
              avg === 1 ? "success" : avg === 0 ? "failure" : variation > 0 ? "mixed" : "shifted"};
        }), ...extra}));
    }
    snapshot("Ready", "Choose a learning profile, then play. Every task starts unseen; the controller cannot see the learner's hidden success rates.");
    for (step = 1; step <= config.steps; step++) {
      excluded = new Set(); retained = []; selected = []; shock = false;
      const stepCounts = CLASSES.map(() => 0), before = totals.candidates;
      let complete = false;
      for (round = 1; round <= (config.mode === "grpo" ? 1 : 4); round++) {
        selected = []; roundStart = totals.candidates;
        const count = config.mode === "grpo" ? 10 : 10 - retained.length;
        due = quota(totals.candidates, count, config.discovery);
        coverageDue = quota(totals.candidates, count, config.coverage);
        totals.coverageReserved += coverageDue;
        const positions = Array.from({length: count}, (_, index) => index);
        for (let i = positions.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [positions[i], positions[j]] = [positions[j], positions[i]];
        }
        const coveragePositions = new Set(positions.slice(0, coverageDue));
        const scores = predictions(tasks, step);
        for (let j = 0; j < count; j++) {
          const reserved = j < due;
          const pick = choose(tasks, scores, excluded, reserved, coveragePositions.has(j), rng);
          if (!pick) {
            snapshot("Stopped", "The step has exhausted its distinct task inventory. No task is repeated to fill the batch.");
            return {config, frames};
          }
          const t = tasks[pick.index];
          excluded.add(t.id); t.seen = true; t.last = totals.candidates;
          totals.candidates++;
          totals.reserved += Number(reserved);
          totals.fulfilled += Number(reserved && pick.newTask);
          totals.coverageFulfilled += Number(pick.covered);
          totals.newTasks += Number(pick.newTask);
          totals.reassessed += Number(pick.reason === "Reassessment");
          stepCounts[pick.c]++;
          selected.push({...pick, outcomes: null, mixed: null, prediction: scores.tasks[pick.index]});
          const detail = pick.newTask ? "unseen task" : "familiar task from an earlier step";
          snapshot("Select", pick.id + " · " + pick.reason + ". " + CLASSES[pick.c] +
            " selected through " + (pick.covered ? "class coverage" : "predicted contrast") +
            "; " + detail + ".", {activeTask: pick.id});
        }
        snapshot("Generate", "Generate four fresh answers for each of " + selected.length +
          " distinct tasks. Student revision " + totals.updates + " stays fixed through all refill rounds.");
        for (const pick of selected) {
          const outcomes = Array.from({length: 4}, () => Number(outcomesRng() < model[pick.index].p));
          const success = outcomes.reduce((s, y) => s + y, 0);
          pick.outcomes = outcomes; pick.mixed = success > 0 && success < 4;
          const record = {step, round, success};
          tasks[pick.index].history.push(record);
          if (pick.newTask) tasks[pick.index].discovery = record;
          totals.attempts += 4;
          if (config.mode === "grpo" || pick.mixed) retained.push(pick);
        }
        const nMixed = selected.filter(p => p.mixed).length;
        snapshot("Observe", nMixed + "/" + selected.length + " groups have mixed outcomes. " +
          (config.mode === "grpo" ? "Vanilla GRPO keeps all groups." :
            "OLMo-style retention has " + retained.length + "/10 useful groups.") +
          " All outcomes update evidence, including constant-reward groups.");
        if (config.mode === "grpo" || retained.length === 10) { complete = true; break; }
      }
      if (!complete) {
        round = 4;
        snapshot("Stopped", "Refill limit reached with " + retained.length +
          "/10 mixed groups. This step performs no model update; generated evidence and cost remain recorded.");
        return {config, frames};
      }
      const practiced = retained.filter(p => p.mixed);
      const beforeUpdate = model.map(m => m.p);
      for (const pick of practiced) {
        const m = model[pick.index];
        m.p += directGain(m.p, m.rate, pick.outcomes, config.preset);
      }
      // Shared skill transfer is an assumption of the synthetic learner.
      for (let i = 0; i < model.length; i++) {
        const c = tasks[i].c;
        const count = practiced.filter(p => p.c === c).reduce((s, p) =>
          s + (config.preset === "signal" ? groupSignal(p.outcomes) : 1), 0);
        if (!(config.preset === "noise" && c === 3)) {
          const rate = config.preset === "hidden" && c === 0 ? 0.001 : 0.006;
          model[i].p += (0.995 - model[i].p) * Math.min(0.08, rate * count);
        }
        if (config.preset === "transfer" && c === 2) {
          const algebra = practiced.filter(p => p.c === 1).length;
          model[i].p += (0.85 - model[i].p) * 0.045 * algebra;
        }
        if (config.preset === "forgetting" && step === 10 && c === 0) {
          model[i].p = Math.max(0.05, model[i].p - 0.5); shock = true;
        }
        model[i].gain = model[i].p - beforeUpdate[i];
      }
      totals.optimized += retained.length; totals.updates++;
      history.push({step, counts: stepCounts, candidates: totals.candidates - before,
        newTasks: totals.newTasks, trueMean: mean(model.map(m => m.p))});
      snapshot("Update", shock ?
        "Synthetic forgetting event: arithmetic's hidden success rates drop. The controller must discover the change through later observations." :
        "Student revision " + totals.updates + ". The toy learner changes after " + practiced.length +
        " mixed groups. " + (config.preset === "signal" ?
          "Learning gains scale with each group's observed contrast. " : "") +
        "Previously selected tasks can return next step.");
    }
    step = config.steps;
    snapshot("Complete", "Simulation complete. Compare task coverage, prediction errors, and generated cost; these synthetic results do not establish an LLM training gain.");
    return {config, frames};
  }
  const api = {CLASSES, PRESETS, normalize, inventory, initialModel, groupSignal,
    directGain, classPosterior, predictions, choose, quota, random, simulate};
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TaskDiscovery = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
