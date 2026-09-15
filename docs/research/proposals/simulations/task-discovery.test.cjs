"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const E = require("./task-discovery-engine.js");
let assertions = 0;
function check(value, message) { assertions++; assert.ok(value, message); }

// Independent quota arithmetic, including request splitting.
for (const percent of [0, 5, 20, 35, 100]) {
  for (let n = 0; n < 80; n++) {
    for (let a = 1; a < 12; a++) {
      const expected = Number((BigInt(n + a) * BigInt(percent)) / 100n -
        (BigInt(n) * BigInt(percent)) / 100n);
      assert.equal(E.quota(n, a, percent), expected);
      assert.equal(E.quota(n, a, percent) + E.quota(n + a, 3, percent),
        E.quota(n, a + 3, percent));
    }
  }
}
assert.equal(E.groupSignal([1, 1, 0, 0]), 1);
assert.equal(E.groupSignal([1, 0, 0, 0]), 0.75);
assert.equal(E.directGain(0.35, 0.14, [1, 1, 1, 1], "signal"), 0);
check(E.directGain(0.35, 0.14, [1, 1, 0, 0], "signal") >
  E.directGain(0.35, 0.14, [1, 0, 0, 0], "signal"), "Signal preset responds to group contrast");
assert.equal(E.directGain(0.35, 0.14, [1, 1, 0, 0], "abilities"),
  E.directGain(0.35, 0.14, [1, 0, 0, 0], "abilities"), "Ordinary baseline learns equally from mixed groups");
check(E.directGain(0.9, 0.14, [1, 1, 0, 0], "signal") <
  E.directGain(0.35, 0.14, [1, 1, 0, 0], "signal"), "Less headroom limits the gain");
const normalModel = E.initialModel("normal");
check(normalModel.every(m => m.p >= .1 && m.p <= .88 && m.rate === .14),
  "Default varies starting abilities while holding learning rate fixed");
for (let c = 0; c < 4; c++) {
  check(new Set(normalModel.slice(c * 16, (c + 1) * 16).map(m => m.p)).size > 1,
    "Every default class contains uneven task abilities");
}
const coldStart = E.simulate({steps: 1}).frames[0];
check(new Set(coldStart.classes.map(c => c.trueMean)).size === 4, "Default class abilities differ");
check(coldStart.tasks.every(t => t.success === null && t.variance === null),
  "Hidden starting abilities are not fabricated observations");
check(new Set(coldStart.tasks.map(t => t.predicted)).size === 1,
  "Cold-start forecasts cannot read unequal hidden abilities");
const abilities = E.initialModel("abilities"), signalModel = E.initialModel("signal");
assert.deepEqual(abilities, signalModel, "Signal and ability presets share their initial conditions");
check(new Set(abilities.map(m => m.p)).size > 10 && abilities.every(m => m.rate === 0.14),
  "Starting abilities vary without changing learning rates");
const speeds = E.initialModel("varied");
assert.deepEqual(speeds.map(m => m.p), abilities.map(m => m.p));
check(new Set(speeds.map(m => m.rate)).size === 3, "Speed preset varies rates on the same task population");
const ordinaryStep = E.simulate({preset: "abilities", steps: 1}).frames;
const signalStep = E.simulate({preset: "signal", steps: 1}).frames;
assert.deepEqual(ordinaryStep.find(f => f.phase === "Observe").picks,
  signalStep.find(f => f.phase === "Observe").picks, "The first groups and outcomes are identical before different learning responses");
check(signalStep.at(-1).tasks.every((t, i) => t.truth <= ordinaryStep.at(-1).tasks[i].truth + 1e-12),
  "Contrast scaling changes the realized learner update");
check(signalStep.at(-1).tasks.some((t, i) => t.truth < ordinaryStep.at(-1).tasks[i].truth - 1e-6),
  "Signal scaling has an observable effect in the seeded first update");

const tasks = E.inventory();
const unknown = E.classPosterior(tasks, 0, 1);
assert.equal(unknown.mean, 0.5);
tasks[0].discovery = {step: 1, success: 2};
const once = E.classPosterior(tasks, 0, 1);
tasks[0].history = Array.from({length: 100}, () => ({step: 1, success: 2}));
assert.deepEqual(E.classPosterior(tasks, 0, 1), once,
  "Repeated groups from one task cannot multiply class sample size");
assert.deepEqual(E.classPosterior(tasks, 0, 10), once,
  "Class discovery evidence remains one observation until an explicit evidence policy discounts it");

const sparse = E.inventory();
for (let i = 0; i < 5; i++) sparse[i].discovery = {step: 1, success: i < 2 ? 2 : 4};
const dense = E.inventory();
for (let i = 0; i < 16; i++) dense[i].discovery = {step: 1, success: i < 2 ? 2 : 4};
check(E.classPosterior(sparse, 0, 1).priority > E.classPosterior(dense, 0, 1).priority,
  "Equal useful counts receive more discovery priority when the sample is smaller");

// The selector gets task identity/evidence only; arbitrary hidden truth cannot influence it.
const scores = E.predictions(tasks, 1);
const a = E.choose(tasks, scores, new Set(), true, false, E.random(5));
const polluted = tasks.map(t => ({...t, hiddenP: 1, truth: 0, rate: 999}));
const b = E.choose(polluted, scores, new Set(), true, false, E.random(5));
assert.deepEqual(a, b);
tasks.forEach((t, i) => { t.seen = true; t.last = i; });
const zeroScores = {classes: [0, 0, 0, 0], tasks: tasks.map(() => 0)};
const recheck = E.choose(tasks, zeroScores, new Set(), false, false, E.random(3));
check(recheck.reason === "Reassessment", "All-zero task scores fall back to reassessment");

for (const preset of Object.keys(E.PRESETS)) {
  for (const mode of ["grpo", "olmo"]) {
    for (const seed of [17, 29]) {
      const {frames} = E.simulate({preset, mode, seed, steps: 18});
      const pickedByStep = new Map(), lifetime = new Set();
      let previousTruth = null, previousUpdates = 0, candidates = 0;
      for (const frame of frames) {
        if (frame.phase === "Select") {
          const pick = frame.picks.at(-1);
          const used = pickedByStep.get(frame.step) || new Set();
          check(!used.has(pick.id), "No repeated task across a step's rounds");
          check(pick.newTask === !lifetime.has(pick.id), "Novelty means unseen in the run");
          used.add(pick.id); pickedByStep.set(frame.step, used); lifetime.add(pick.id);
          candidates++;
          check(pick.chance > 0 && pick.chance <= 1, "Valid conditional selection probability");
          if (pick.reserved && !pick.newTask) check(lifetime.size === 64, "Only exhausted discovery may reuse tasks");
        }
        if (frame.phase === "Generate") {
          assert.equal(frame.totals.reserved, Math.floor(frame.totals.candidates * 0.2));
          assert.equal(frame.totals.coverageReserved, Math.floor(frame.totals.candidates * 0.2));
          assert.equal(frame.totals.coverageFulfilled, frame.totals.coverageReserved);
        }
        assert.equal(frame.totals.candidates, candidates);
        assert.equal(frame.totals.newTasks, lifetime.size);
        const truth = frame.tasks.map(t => t.truth);
        if (previousTruth && previousUpdates === frame.totals.updates) {
          assert.deepEqual(truth, previousTruth, "Learner stays fixed between updates, including refills");
        }
        previousTruth = truth; previousUpdates = frame.totals.updates;
        const active = frame.classes.filter(c => c.unseen);
        if (active.length) check(Math.abs(active.reduce((s, c) => s + c.discoveryShare, 0) - 1) < 1e-10, "Class shares sum to one");
        check(truth.every(p => p >= 0 && p <= 1), "Hidden learner probabilities stay valid");
      }
      const final = frames.at(-1);
      check(final.totals.attempts === final.totals.candidates * 4, "All generated candidates count toward cost");
      if (preset === "solved" && mode === "olmo") {
        assert.equal(final.phase, "Stopped");
        assert.equal(final.totals.updates, 0);
        assert.equal(final.totals.candidates, 40);
      }
      if (preset === "noise") {
        check(final.tasks.filter(t => t.c === 3).every(t => t.truth === 0.5),
          "Irreducible noise can attract variance priority without learning");
      }
    }
  }
}
const full = E.simulate({steps: 30}).frames.at(-1);
assert.equal(full.totals.newTasks, 64);
check(full.totals.reserved > full.totals.fulfilled, "Exhaustion is visible, not relabeled novelty");
for (const discovery of [0, 100]) {
  const f = E.simulate({discovery, steps: 8}).frames.at(-1);
  assert.equal(f.totals.reserved, discovery === 0 ? 0 : 80);
}
assert.deepEqual(E.simulate({steps: 2}), E.simulate({steps: 2}), "Seeded replay is deterministic");
console.log("Engine checks passed across " + Object.keys(E.PRESETS).length * 4 +
  " scenario/mode/seed runs plus budget boundaries (" + assertions + " checks).");

if (process.argv.includes("--dom")) {
  const {JSDOM, VirtualConsole} = require("jsdom");
  for (const file of ["task-discovery-v2.html", "../configurable-curriculum-and-data-preparation-v2.html"]) {
    const errors = [], virtualConsole = new VirtualConsole();
    virtualConsole.on("jsdomError", e => errors.push(e));
    let tick = null;
    const dom = new JSDOM(fs.readFileSync(path.join(__dirname, file), "utf8"), {
      runScripts: "dangerously", pretendToBeVisual: true, virtualConsole,
      beforeParse(w) {
        w.setInterval = fn => { tick = fn; return 1; };
        w.clearInterval = () => { tick = null; };
      },
    });
    const d = dom.window.document, el = id => d.getElementById("td-" + id);
    check(errors.length === 0, "No DOM/script startup errors: " + errors.map(e => e.message).join(", "));
    assert.equal(el("preset").value, "normal");
    assert.equal(el("preset").querySelectorAll('optgroup[label="Ordinary learning"] option').length, 4);
    assert.equal(el("preset").querySelectorAll('optgroup[label="Stress cases"] option').length, 5);
    assert.equal(d.querySelectorAll(".td-task").length, 64);
    assert.equal(d.querySelectorAll(".td-success-history").length, 4);
    assert.equal(d.querySelectorAll(".td-allocation-segment,.td-success-point").length, 0,
      "Future evidence is hidden before playback");
    el("play").click(); assert.equal(el("play").textContent, "Pause");
    tick(); check(d.querySelectorAll(".td-group").length === 10, "Selection reveals the entire request before generation");
    assert.equal(d.querySelectorAll(".td-lane").length, 3);
    assert.equal(d.querySelector('[data-lane="discovery"]').dataset.share, "1", "Startup actual discovery is 100%");
    check(el("lane-note").textContent.includes("2/2 slots filled (20% cumulative target)"),
      "Startup actual share stays distinct from the discovery reserve");
    assert.equal(d.querySelectorAll(".td-task.selected").length, 10);
    tick(); assert.equal(d.querySelector(".td-training").dataset.phase, "Generate");
    assert.equal(d.querySelectorAll(".td-group").length, 10, "Generate keeps the complete lanes");
    el("play").click(); assert.equal(el("play").textContent, "Play");
    el("step").click(); assert.equal(el("revision").textContent, "Student 1");
    assert.equal(d.querySelectorAll(".td-group").length, 10);
    assert.equal(d.querySelectorAll(".td-answer.pass,.td-answer.fail").length >= 40, true);
    const segments = [...d.querySelectorAll(".td-allocation-segment")];
    check(Math.abs(segments.reduce((sum, node) => sum + Number(node.dataset.share), 0) - 1) < 1e-10,
      "First allocation column totals one");
    check(segments.every(node => node.dataset.step === "1"), "Allocation shows no future steps");
    const firstObserved = E.simulate({steps: 1}).frames.find(f => f.phase === "Observe");
    assert.equal(d.querySelectorAll(".td-success-point").length,
      firstObserved.classes.filter(c => c.success != null).length,
      "Success charts show only classes with observed evidence");
    el("next").click();
    const secondSelection = E.simulate({steps: 2}).frames.filter(f => f.phase === "Select" && f.step === 2).at(-1);
    assert.deepEqual([...d.querySelectorAll(".td-group")].map(g => g.dataset.taskId).sort(),
      secondSelection.picks.map(p => p.id).sort(), "Grouping changes presentation, not selected identities");
    check(Number(d.querySelector('[data-lane="discovery"]').dataset.count) >= 2,
      "The 20% discovery reserve is a floor and adaptive selection may add unseen tasks");
    const laneCounts = [...d.querySelectorAll(".td-lane")].map(l => Number(l.dataset.count));
    assert.equal(laneCounts.reduce((a, b) => a + b, 0), 10);
    check(Math.abs([...d.querySelectorAll(".td-lane")].reduce((s, l) => s + Number(l.dataset.share), 0) - 1) < 1e-10,
      "Realized lane shares sum to one");
    d.querySelector('[data-task="G16"]').click();
    check(el("inspector").textContent.includes("G16"), "Task inspector follows selection");
    el("truth").checked = true; el("truth").dispatchEvent(new dom.window.Event("change"));
    check(el("inspector").textContent.includes("Hidden solve probability"), "Truth is an optional inspection view");
    check(el("inspector").textContent.includes("Last model update"), "Inspection exposes the assumed learning gain");
    el("replay").value = 0; el("replay").dispatchEvent(new dom.window.Event("input"));
    assert.equal(el("revision").textContent, "Student 0");
    assert.equal(d.querySelectorAll(".td-group").length, 0);
    assert.equal(d.querySelectorAll(".td-allocation-segment,.td-success-point").length, 0,
      "Scrubbing backwards removes future chart history");
    el("preset").value = "signal"; el("preset").dispatchEvent(new dom.window.Event("change"));
    check(el("scenario").textContent.includes("scale with observed group contrast"), "Signal behavior is explained inline");
    el("step").click(); assert.equal(el("revision").textContent, "Student 1");
    el("preset").value = "solved"; el("mode").value = "olmo";
    el("mode").dispatchEvent(new dom.window.Event("change"));
    el("step").click();
    check(el("narration").textContent.includes("no model update"), "Bounded all-success refill failure is explained");
    assert.equal(el("position").textContent, "Step 1 · Round 4");
    assert.equal(d.querySelectorAll(".td-task.used").length, 40, "Earlier refill picks remain excluded in the pool");
    assert.equal(d.querySelectorAll(".td-task.selected").length, 10, "Outline distinguishes current round from earlier rounds");
    el("reset").click(); assert.equal(el("revision").textContent, "Student 0");
    assert.equal(d.querySelectorAll(".td-task.unseen").length, 64);
    // Hand-built visual fixtures distinguish identical success rates with different
    // within-group variance. Unknown evidence must never look like observed zero.
    const simulate = dom.window.TaskDiscovery.simulate;
    const fixture = simulate({steps: 1}).frames[0];
    [[1, 0, "success"], [0, 0, "failure"], [.5, 1, "mixed"], [.5, 0, "shifted"], [null, null, "uncertain"]]
      .forEach(([success, variance, state], i) => Object.assign(fixture.tasks[i], {success, variance, state, seen: true}));
    dom.window.TaskDiscovery.simulate = () => ({frames: [fixture]});
    el("reset").click();
    const tiles = [...d.querySelectorAll(".td-task")].slice(0, 5);
    assert.equal(tiles[0].style.getPropertyValue("--td-fill"), tiles[1].style.getPropertyValue("--td-fill"),
      "All-success and all-failure tasks have equal colors because both have zero variance");
    assert.notEqual(tiles[2].style.getPropertyValue("--td-fill"), tiles[3].style.getPropertyValue("--td-fill"),
      "Different variance maps to different colors despite equal success");
    assert.equal(tiles[0].querySelector(".td-success i").style.width, "100%");
    assert.equal(tiles[1].querySelector(".td-success i").style.width, "0%");
    assert.equal(tiles[2].querySelector(".td-success i").style.width, "50%");
    assert.equal(tiles[3].querySelector(".td-success i").style.width, "50%");
    assert.equal(tiles[4].querySelector(".td-success"), null);
    assert.equal(tiles[4].dataset.variance, "unknown");
    check(tiles[2].getAttribute("aria-label").includes("success 50.0%; within-group variance 1.00"),
      "Numeric evidence is available without relying on color");
    dom.window.TaskDiscovery.simulate = simulate;
    el("reset").click();
    const ids = [...d.querySelectorAll("[id]")].map(x => x.id);
    assert.equal(ids.length, new Set(ids).size, "No duplicate DOM IDs");
    check(errors.length === 0, "No DOM errors after interaction");
    dom.window.close();
  }
  console.log("Standalone and embedded DOM checks passed: play/pause, next step, inspection, truth toggle, scrub, reset, and refill failure.");
}
