(function () {
  "use strict";
  const E = globalThis.TaskDiscovery;
  const el = id => document.getElementById("td-" + id);
  const percent = n => n == null ? "—" : (n * 100).toFixed(1) + "%";
  let replay, position = 0, timer = null, inspected = null, renderedPosition = -1;
  // Variance color and observed success length use independent scales.
  function varianceColor(value) {
    if (value == null) return "#efedf0";
    const stops = [[242, 211, 180], [249, 247, 239], [169, 211, 190]];
    const half = value < .5 ? 0 : 1, fraction = value * 2 - half;
    return "rgb(" + stops[half].map((v, i) => Math.round(v +
      fraction * (stops[half + 1][i] - v))).join(",") + ")";
  }
  const descriptions = {
    normal: "Tasks start with different solve probabilities, from 10% to 88%, varying within and across classes. Useful practice improves them, with smaller gains transferring within a class. These starting abilities are hidden from the controller.",
    abilities: "Tasks start between 15% and 88% solve probability, with one shared learning rate. The same model can succeed often on one task and rarely on another.",
    signal: "The same starting tasks as Different starting abilities, but learning gains scale with observed group contrast: 2/4 successes gives signal 1; 1/4 or 3/4 gives 0.75; 0/4 or 4/4 gives no direct gain.",
    hidden: "Arithmetic has twelve almost-always-solved tasks and four less-solved tasks. The controller must find those exceptions.",
    varied: "The same starting abilities, but tasks respond at slow, medium, or fast learning rates. Repeated useful practice changes their solve probabilities at different speeds.",
    transfer: "Geometry initially almost always fails. Algebra practice gradually makes geometry more learnable, including tasks not yet selected.",
    forgetting: "After model update 10, arithmetic success rates drop. Only fresh outcomes reveal that change to the controller.",
    noise: "Probability tasks stay at 50% success regardless of practice. Their persistent variance can attract compute without producing learning.",
    solved: "Every task always succeeds. GRPO can proceed with zero outcome contrast; OLMo-style mode stops when its refill bound is reached.",
  };
  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null; el("play").textContent = "Play";
  }
  function schedule() {
    stop(); el("play").textContent = "Pause";
    timer = setInterval(() => {
      if (position >= replay.frames.length - 1) { stop(); return; }
      position++; render();
    }, Number(el("speed").value));
  }
  function reset() {
    stop(); position = 0; inspected = null; renderedPosition = -1;
    const seed = Math.min(9999, Math.max(1, Math.trunc(Number(el("seed").value) || 17)));
    el("seed").value = seed;
    const config = {preset: el("preset").value, mode: el("mode").value, seed,
      coverage: Number(el("coverage").value), discovery: Number(el("discovery").value)};
    el("coverage-value").textContent = config.coverage + "%";
    el("discovery-value").textContent = config.discovery + "%";
    replay = E.simulate(config);
    // Preserve sequential conditional draws in the engine; present whole requests.
    replay.frames = replay.frames.filter((frame, i, frames) =>
      frame.phase !== "Select" || frames[i + 1]?.phase !== "Select");
    el("replay").max = replay.frames.length - 1;
    el("scenario").textContent = descriptions[config.preset] +
      (config.coverage === 0 ? " Zero class exploration disables coverage-driven reassessment." : "") +
      (config.discovery === 0 ? " Zero discovery disables the protected unseen-task allocation after startup." : "");
    render();
  }
  const dots = outcomes => '<div class="td-outcomes">' +
    (outcomes || [null, null, null, null]).map(y => '<i class="td-answer ' +
      (y === 1 ? "pass" : y === 0 ? "fail" : "") + '"></i>').join("") + "</div>";
  const outcomesFrom = success => Array.from({length: 4}, (_, i) => Number(i < success));
  function inspector(f) {
    const id = inspected || (f.picks[0] && f.picks[0].id) || "A01";
    const t = f.tasks.find(x => x.id === id);
    const pick = f.picks.find(x => x.id === id);
    const last = t.recent.length ? t.recent[t.recent.length - 1] : null;
    const own = t.recent.length ? "Class prior + " + t.recent.length + " recent task groups." :
      t.seen ? "No fresh outcomes. Using its class prior." : "Unseen: provisional class-based prediction.";
    el("inspector").innerHTML = "<h4>" + t.id + " <span class=\"td-small\">" + E.CLASSES[t.c] +
      '</span></h4><p class="td-small">' + own + "</p><dl><dt>Predicted useful group</dt><dd>" +
      percent(t.predicted) + "</dd><dt>Observed success</dt><dd>" + percent(t.success) +
      "</dd><dt>Observed variance¹</dt><dd>" + (t.variance == null ? "—" : t.variance.toFixed(2)) +
      "</dd><dt>Lifetime groups</dt><dd>" + t.groups + "</dd><dt>Last fresh evidence</dt><dd>" +
      (last ? "Step " + last.step : "—") + "</dd></dl>" +
      (pick ? '<p class="td-small">Selected: ' + pick.reason + "<br>Class route: " +
        (pick.covered ? "coverage" : "predicted contrast") + "<br>Draw probability: " + percent(pick.chance) + "</p>" : "") +
      (t.recent.length ? '<div class="td-small">Recent groups</div>' + t.recent.map(h =>
        '<div class="td-history-row"><span>Step ' + h.step + " · r" + h.round + "</span>" +
        dots(outcomesFrom(h.success)) + "</div>").join("") : '<p class="td-small">No observed attempts yet.</p>') +
      '<p class="td-small">¹ 4p̄(1 − p̄), averaged over recent groups. Zero can mean all successes or all failures.</p>' +
      (el("truth").checked ? '<div class="td-truth">Hidden solve probability: ' + percent(t.truth) +
        "<br>Last model update: " + (t.lastGain >= 0 ? "+" : "") + (t.lastGain * 100).toFixed(2) +
        " percentage points<br>Inspection only; unavailable to the sampler.</div>" : "");
  }
  function timelines(f) {
    const colors = ["#7255bb", "#207f89", "#ac6532", "#657b42"];
    const steps = replay.config?.steps || f.config.steps || 30;
    const bins = new Map();
    // Derive only from events already reached, including incomplete refill steps.
    for (const event of replay.frames.slice(0, position + 1)) {
      if (["Ready", "Complete"].includes(event.phase)) continue;
      const bin = bins.get(event.step) || {step: event.step, counts: [0, 0, 0, 0], success: null};
      bin.counts = [0, 0, 0, 0];
      event.used.forEach(id => { bin.counts[event.tasks.find(t => t.id === id).c]++; });
      if (event.phase === "Observe") bin.success = event.classes.map(c => c.success);
      bins.set(event.step, bin);
    }
    const data = [...bins.values()].sort((a, b) => a.step - b.step);
    const left = 36, width = 590, slot = width / steps;
    const x = step => left + (step - .5) * slot;
    const cursor = data.length ? x(data.at(-1).step) : left;
    const grid = height => [0, .5, 1].map(v =>
      '<line x1="36" x2="626" y1="' + (12 + (1 - v) * height) + '" y2="' +
      (12 + (1 - v) * height) + '" stroke="#e5e0e7"/><text x="30" y="' +
      (15 + (1 - v) * height) + '" text-anchor="end">' + v * 100 + '%</text>').join("");
    const axis = y => '<text x="36" y="' + y + '">Step 1</text><text x="626" y="' + y +
      '" text-anchor="end">Step ' + steps + '</text>';
    const marker = height => '<line x1="' + cursor + '" x2="' + cursor +
      '" y1="10" y2="' + (height + 14) + '" stroke="#514559" stroke-dasharray="3 3"/>';
    el("chart-legend").innerHTML = E.CLASSES.map((name, i) =>
      '<span><i style="background:' + colors[i] + '"></i>' + name + '</span>').join("");
    el("allocation-chart").innerHTML = '<svg viewBox="0 0 640 126" role="img" aria-label="Class allocation timeline: actual candidate shares by training step">' +
      grid(80) + data.map(bin => {
        const total = bin.counts.reduce((a, b) => a + b, 0);
        let bottom = 92;
        return bin.counts.map((count, c) => {
          if (!count || !total) return "";
          const height = 80 * count / total; bottom -= height;
          return '<rect class="td-allocation-segment" data-step="' + bin.step + '" data-share="' +
            count / total + '" x="' + (x(bin.step) - slot / 2 + .5) + '" y="' + bottom +
            '" width="' + (slot - 1) + '" height="' + height + '" fill="' + colors[c] +
            '"><title>Step ' + bin.step + ' · ' + E.CLASSES[c] + ': ' + count + '/' + total +
            ' candidates (' + percent(count / total) + ')</title></rect>';
        }).join("");
      }).join("") + marker(80) + axis(115) + '</svg>';
    el("success-charts").innerHTML = E.CLASSES.map((name, c) => {
      let path = "", connected = false;
      const points = data.map(bin => {
        const value = bin.success?.[c];
        if (value == null) { connected = false; return ""; }
        const y = 12 + (1 - value) * 44;
        path += (connected ? " L" : " M") + x(bin.step) + "," + y;
        connected = true;
        return '<circle class="td-success-point" data-step="' + bin.step + '" data-success="' +
          value + '" cx="' + x(bin.step) + '" cy="' + y + '" r="2.3" fill="' + colors[c] +
          '"><title>Step ' + bin.step + ' · ' + name + ': ' + percent(value) + ' recent observed success</title></circle>';
      }).join("");
      const latest = [...data].reverse().find(bin => bin.success);
      return '<div class="td-success-history"><div class="td-history-heading"><strong style="color:' +
        colors[c] + '">' + name + '</strong><span>Latest observed: ' + percent(latest?.success[c]) +
        '</span></div><svg viewBox="0 0 640 82" role="img" aria-label="' + name +
        ' recent observed success from zero to one hundred percent">' + grid(44) +
        '<path d="' + path + '" fill="none" stroke="' + colors[c] + '" stroke-width="2"/>' +
        points + marker(44) + axis(77) + '</svg></div>';
    }).join("");
  }
  function render() {
    const f = replay.frames[position], end = position === replay.frames.length - 1;
    const previous = renderedPosition >= 0 && position === renderedPosition + 1 ?
      replay.frames[renderedPosition] : null;
    el("replay").value = position;
    el("revision").textContent = "Student " + f.totals.updates;
    el("revision").classList.toggle("td-updating", f.phase === "Update");
    const training = document.querySelector("#task-discovery-player .td-training");
    training.dataset.phase = f.phase;
    training.classList.toggle("td-just-selected", f.phase === "Select" && renderedPosition !== position);
    el("position").textContent = "Step " + f.step + " · Round " + f.round;
    el("next").disabled = end; el("step").disabled = end;
    const missing = f.totals.reserved - f.totals.fulfilled;
    el("statistics").innerHTML = [
      [f.totals.newTasks + "/64", "Unique tasks discovered"],
      [f.totals.fulfilled + "/" + f.totals.reserved, "Reserved discovery filled"],
      [f.totals.candidates, "Candidate groups · " + f.totals.attempts + " attempts"],
      [f.totals.optimized, "Groups used in completed updates"],
    ].map(x => '<div class="td-stat"><strong>' + x[0] + "</strong><span>" + x[1] + "</span></div>").join("");
    el("pool").innerHTML = f.classes.map((c, ci) => {
      const recent = f.history.slice(-15);
      const bars = recent.map((h, i) => {
        const height = 20 * h.counts[ci] / h.candidates;
        return '<rect x="' + (i * 6.5) + '" y="' + (22 - height) + '" width="4.5" height="' + height +
          '"><title>Step ' + h.step + ": " + h.counts[ci] + "/" + h.candidates + " candidates</title></rect>";
      }).join("");
      return '<div class="td-class"><div class="td-class-name">' + c.name + '</div><span class="td-class-count">' +
        c.unseen + '/16 unseen</span><div class="td-class-forecast">Useful-task estimate <b>' + percent(c.prediction) +
        '</b> · uncertainty <b>' + percent(c.uncertainty) +
        '</b><br>Discovery index <b>' + percent(c.priority) +
        '</b><br>Next discovery share <b>' + (c.unseen ? percent(c.discoveryShare) : "ineligible") +
        '</b></div><div class="td-meter"><i style="width:' + c.discoveryShare * 100 + '%"></i></div><div class="td-task-grid">' +
        f.tasks.filter(t => t.c === ci).map(t => {
          const selected = f.picks.some(p => p.id === t.id), used = f.used.includes(t.id);
          const state = !t.seen ? "Unseen" : t.success == null ? "No recent evidence" :
            "Observed success " + percent(t.success) + "; within-group variance " + t.variance.toFixed(2);
          const before = previous?.tasks.find(x => x.id === t.id)?.success;
          const from = before ?? t.success;
          return '<button type="button" class="td-task ' + t.state + (used ? " used" : "") +
            (selected ? " selected" : "") + '" style="--td-fill:' + varianceColor(t.variance) +
            '" data-success="' + (t.success ?? "unknown") + '" data-variance="' + (t.variance ?? "unknown") +
            '" data-task="' + t.id + '" aria-pressed="' +
            (inspected === t.id) + '" title="' + t.id + ": " + state + "; useful-group forecast " +
            percent(t.predicted) + "; " + t.groups + ' observed groups" aria-label="' +
            t.id + ", " + state + (selected ? ", selected this round" : used ? ", used earlier this step" : "") +
            ', inspect task">' + t.id + (t.success == null ? '<span class="td-unknown">—</span>' :
              '<span class="td-success" aria-hidden="true"><i style="--td-from:' + from * 100 +
              '%;width:' + t.success * 100 + '%"></i></span>') + '</button>';
        }).join("") + '</div><svg class="td-spark" viewBox="0 0 100 24" role="img" aria-label="' +
        c.name + ' candidate share over the last fifteen completed steps">' + bars +
        '</svg><span class="td-spark-label">Actual candidate share · recent steps</span>' +
        (el("truth").checked ? '<div class="td-truth">Hidden class success ' + percent(c.trueMean) + "</div>" : "") + "</div>";
    }).join("");
    inspector(f);
    el("stages").innerHTML = ["Select", "Generate", "Observe", "Update"].map((p, i) =>
      '<span class="td-stage ' + (p === f.phase ? "active" : "") + '">' + (i + 1) + " · " + p + "</span>").join("");
    el("request").textContent = f.phase === "Ready" ? "Ten distinct tasks · four answers each. Play to fill the lanes." :
      "Round " + f.round + " · " + f.picks.length + " tasks · actual shares of this request";
    const lanes = [
      {key: "discovery", name: "Discovery", hint: "First attempt", reasons: ["Discovery", "Additional new task"]},
      {key: "practice", name: "Practice", hint: "Predicted contrast", reasons: ["Adaptive practice"]},
      {key: "recheck", name: "Recheck", hint: "Revisit familiar tasks", reasons: ["Reassessment"]},
    ];
    el("tray").innerHTML = lanes.map((lane, index) => {
      const picks = f.picks.filter(p => lane.reasons.includes(p.reason));
      const share = f.picks.length ? picks.length / f.picks.length : null;
      return '<section class="td-lane" data-lane="' + lane.key + '" data-count="' + picks.length +
        '" data-share="' + (share ?? "unknown") + '"><header class="td-lane-label"><strong>' +
        lane.name + '</strong><span class="td-lane-percent">' + percent(share) +
        '</span><small>' + picks.length + ' tasks · ' + lane.hint + '</small></header>' +
        '<div class="td-lane-tasks" style="--td-delay:' + index * 100 + 'ms">' +
        (picks.length ? picks.map(p => '<div class="td-group ' +
          (p.mixed === false && f.config.mode === "olmo" ? "constant" : "") +
          '" data-task-id="' + p.id + '" title="' + p.reason + " · " +
          (p.covered ? "Class coverage" : "Adaptive class draw") +
          " · Conditional task probability " + percent(p.chance) +
          (p.outcomes ? " · " + p.outcomes.join(", ") : "") + '">' +
          "<strong>" + p.id + '</strong>' + dots(p.outcomes) +
          (p.reason === "Discovery" ? '<span class="td-tag">reserved</span>' : "") +
          '</div>').join("") : '<span class="td-lane-empty">' +
            (f.phase === "Ready" ? "Awaiting selection" : "No tasks this round") + '</span>') + '</div></section>';
    }).join("");
    const reservedPicks = f.picks.filter(p => p.reason === "Discovery").length;
    el("lane-note").textContent = f.phase === "Ready" ?
      "Discovery reserves " + f.config.discovery + "% across requests. Practice and recheck shares follow the sampler." :
      "Discovery reserve: " + reservedPicks + "/" + f.due + " slots filled (" + f.config.discovery +
      "% cumulative target). Class coverage: " + f.picks.filter(p => p.covered).length + "/" +
      f.coverageDue + " routed through the cumulative coverage ledger. " +
      (f.picks.some(p => p.reason === "Additional new task") ?
        "Additional new tasks fill startup or unavailable familiar-task slots. " : "") +
      "Recheck is an observed share, not a separate quota. " + f.used.length + " task IDs excluded until the next step.";
    el("flow").textContent = f.phase === "Generate" ? "Generate answers → verify → return evidence" :
      f.phase === "Observe" ? (f.config.mode === "olmo" ? f.retained + "/10 mixed groups retained across rounds. " : "") +
      "Fresh evidence is now available to the next selection." :
      f.phase === "Update" ? "Model updated → clear step exclusions → select the next batch" :
      f.phase === "Stopped" ? "No model update for this incomplete step." : "Class choice → task choice → fresh group";
    const message = f.phase === "Select" ? "The request is selected: " +
      lanes.map(lane => f.picks.filter(p => lane.reasons.includes(p.reason)).length + " " + lane.name.toLowerCase()).join(", ") +
      ". Generate four fresh answers for every task, then return the outcomes to the controller." : f.message;
    el("narration").textContent = message + (missing ? " Unseen inventory exhausted: " + missing +
      " reserved slots have become reassessment, not new tasks." : "");
    timelines(f);
    renderedPosition = position;
    if (end) stop();
  }
  el("play").addEventListener("click", () => {
    if (timer !== null) stop();
    else { if (position === replay.frames.length - 1) position = 0; schedule(); render(); }
  });
  el("next").addEventListener("click", () => { stop(); position = Math.min(position + 1, replay.frames.length - 1); render(); });
  el("step").addEventListener("click", () => {
    stop();
    do { position++; } while (position < replay.frames.length - 1 &&
      !["Update", "Stopped", "Complete"].includes(replay.frames[position].phase));
    position = Math.min(position, replay.frames.length - 1); render();
  });
  el("reset").addEventListener("click", reset);
  el("replay").addEventListener("input", () => { stop(); position = Number(el("replay").value); render(); });
  el("pool").addEventListener("click", e => {
    const task = e.target.closest("[data-task]");
    if (task) { inspected = task.dataset.task; render(); }
  });
  el("truth").addEventListener("change", render);
  el("speed").addEventListener("change", () => { if (timer !== null) schedule(); });
  ["preset", "mode", "seed", "coverage", "discovery"].forEach(id => el(id).addEventListener("change", reset));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop(); });
  reset();
})();
