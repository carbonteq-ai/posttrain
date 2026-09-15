"""Render the v2 reading report and embed the unchanged technical specification.

From the repository root:
  uv run --no-project --with pypandoc-binary --with beautifulsoup4 \
    python docs/research/proposals/render_curriculum_v2.py
"""

import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pypandoc
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
STEM = "configurable-curriculum-and-data-preparation"

CSS = """
.math-display{background:transparent;border:0;border-radius:0;box-shadow:none;padding:12px 0;margin:26px 0;text-align:center}
article{--reading-width:44rem}
article>p{font:inherit;color:var(--ink);max-width:var(--reading-width);margin:16px 0}
article>p:has(>em:only-child){font:inherit;color:var(--ink);max-width:var(--reading-width);margin:16px 0}
article>.math-display{width:100%;max-width:var(--reading-width);font-size:21px;padding:6px 0;margin:22px 0;line-height:1.5}
article>.math-display math{font-size:inherit;math-style:normal}
article>.math-display mtd{padding:8px 4px;math-style:normal}
article>.math-display mtext{font-size:.85em}
article>.table-wrap{max-width:var(--reading-width);border:0;border-radius:0;background:transparent}
article>.table-wrap table{font-size:14px;background:transparent}
article>.table-wrap th{background:transparent;border-top:1px solid var(--line);color:var(--ink)}
article>.table-wrap td,article>.table-wrap th{padding:12px 10px}
.calculation-example{max-width:var(--reading-width);margin:32px 0}
.calculation-example h4{font:600 16px/1.5 var(--sans);margin:0 0 8px}
.calculation-example p{font:inherit;color:var(--ink);margin:12px 0}
.calculation-example p:has(>em:only-child){font:inherit;color:var(--ink);margin:12px 0}
.calculation-example .table-wrap{border:0;border-radius:0;background:transparent;margin:18px 0}
.calculation-example table{font:14px/1.65 var(--sans);min-width:540px;font-variant-numeric:tabular-nums}
.calculation-example caption{text-align:left;color:var(--secondary);font-size:13px;padding:0 0 12px}
.calculation-example th{background:transparent;border-top:1px solid var(--line);color:var(--ink)}
.calculation-example th,.calculation-example td{padding:11px 10px;text-align:right;white-space:nowrap}
.calculation-example th:first-child,.calculation-example td:first-child{text-align:left}
.calculation-example td:last-child{font-weight:600;color:var(--ink)}
.calculation-example tbody th{font-weight:500;border-top:0}
.calculation-example tbody tr:last-child td,.calculation-example tbody tr:last-child th{border-bottom:1px solid var(--line)}
.calculation-example details{margin-top:16px;padding-top:14px;border-top:1px solid var(--line)}
.calculation-example summary{cursor:pointer;color:var(--accent);font-weight:500}
.calculation-example .math-display{font-size:20px}
.calculation-example mtd{padding:8px 4px;math-style:normal}
@media(max-width:650px){article>.math-display{font-size:19px}.calculation-example table{font-size:13px}.calculation-example .table-wrap:focus-visible{outline:2px solid var(--accent);outline-offset:3px}}
.reader-reference,.reader-demo{margin:26px 0;padding:18px 20px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.reader-reference>summary,.reader-demo>summary{cursor:pointer;font-weight:600;color:var(--accent);line-height:1.6}
.reader-reference[open]>summary,.reader-demo[open]>summary{margin-bottom:24px}
article>p,article>ul,article>ol{max-width:var(--reading-width)}
article>p:first-of-type{font-size:12px}
@media print{.reader-reference,.reader-demo{border:0;padding:0}}
.architecture-v2{margin:28px 0 12px;padding:22px;border:1px solid var(--line);border-radius:9px;background:var(--surface)}
.architecture-v2 .inventory{padding:12px 15px;background:var(--subtle);border:1px solid var(--line);border-radius:6px}
.architecture-v2 .caption{display:block;font-size:11px;color:var(--secondary);line-height:1.55;margin-top:5px}
.architecture-v2 strong{font-size:13px}
.architecture-v2 .stage-label{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--accent);margin-bottom:8px}
.architecture-v2 .parts{display:grid;grid-template-columns:1fr 34px 1.3fr;align-items:stretch;gap:10px;margin:18px 0}
.architecture-v2 .controller,.architecture-v2 .training{padding:16px;border:1px solid var(--line);border-radius:7px}
.architecture-v2 .controller{background:#f3effb;border-color:#ded5ef}
.architecture-v2 .arrow{align-self:center;text-align:center;color:var(--accent);font-size:24px}
.architecture-v2 .choices{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}
.architecture-v2 .choice{padding:10px;background:var(--subtle);border-radius:5px}
.architecture-v2 .pipeline{padding:10px 0 0;border-top:1px solid var(--line);font-size:12px}
.architecture-v2 .feedback{padding:12px 15px;border-radius:6px;background:#edf2ee;font-size:12px}
.architecture-v2 .inventory span{display:inline-block;font-size:11px;color:var(--secondary);margin-right:16px}
.forecast-v2{margin:26px 0;padding:20px;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
.forecast-v2 h4{font-size:16px;margin:0 0 8px}
.forecast-v2 p{font-size:12px;color:var(--secondary);margin:8px 0}
.forecast-v2 label{display:block;font-size:13px;margin-top:18px}
.forecast-v2 input{display:block;width:100%;accent-color:var(--accent);margin:12px 0}
.forecast-v2 .forecast-values{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
.forecast-v2 .forecast-values strong{font-size:23px;display:block;font-variant-numeric:tabular-nums}
.forecast-v2 .meter{background:var(--subtle);height:10px;border-radius:4px;overflow:hidden;margin:9px 0}
.forecast-v2 .meter span{display:block;background:#a99ad2;height:100%;transition:width .15s}
.forecast-v2 .baseline{border-top:1px solid var(--line);padding-top:12px;margin-top:16px}
@media(max-width:650px){.architecture-v2{padding:14px}.architecture-v2 .parts{grid-template-columns:1fr}.architecture-v2 .arrow{transform:rotate(90deg)}.forecast-v2 .forecast-values{grid-template-columns:1fr 1fr}}
@media(prefers-reduced-motion:reduce){.forecast-v2 .meter span{transition:none}}
"""

ARCHITECTURE = """
<figure class="architecture-v2" aria-label="Task discovery and practice around the training loop">
  <div class="inventory">
    <div class="stage-label">Task inventory</div>
    <span>Arithmetic · A1, A2, …</span><span>Algebra · B1, B2, …</span>
    <span>Each ID is unseen or previously selected</span>
  </div>
  <div class="parts">
    <div class="controller">
      <div class="stage-label">Persistent controller</div>
      <strong>Predict the next useful group</strong>
      <span class="caption">Class evidence gives unseen tasks a provisional score. Each task's fresh outcomes refine its own score.</span>
      <span class="caption">Retain discovery count, seen IDs, current-step exclusions, and observation history.</span>
    </div>
    <div class="arrow" aria-hidden="true">→</div>
    <div class="training">
      <div class="stage-label">One optimizer step</div>
      <div class="choices">
        <div class="choice"><strong>Discovery slots</strong><span class="caption">Unseen tasks from promising classes, with class coverage.</span></div>
        <div class="choice"><strong>Other practice</strong><span class="caption">Useful familiar tasks and occasional reassessment.</span></div>
      </div>
      <div class="pipeline">Distinct task groups → fresh answers → verification → algorithm retention and update</div>
      <span class="caption">OLMo refills can use fresh evidence before the update. All rounds share the same task exclusions.</span>
    </div>
  </div>
  <div class="feedback">Completed candidate evidence returns to the controller, including rejected groups. After the model update, familiar tasks become eligible again.</div>
</figure>
"""

FORECAST = """
<section class="forecast-v2" aria-labelledby="forecast-title">
  <h4 id="forecast-title">How a class prediction redirects discovery</h4>
  <p>Two classes have equal base weights. Every unsolved task in this toy population succeeds half the time. Algebra stays 20% solved; each group has four attempts.</p>
  <label for="solved-fraction">Arithmetic tasks that always succeed: <output id="solved-value" for="solved-fraction">60%</output></label>
  <input id="solved-fraction" type="range" min="0" max="100" value="60" step="5" aria-describedby="forecast-assumption">
  <div class="forecast-values" aria-live="polite">
    <div><p>Arithmetic · predicted mixed groups</p><strong id="mixed-value">35.0%</strong><div class="meter"><span id="mixed-bar" style="width:35%"></span></div></div>
    <div><p>Arithmetic · share of discovery draws</p><strong id="discovery-value">36.7%</strong><div class="meter"><span id="discovery-bar" style="width:36.7%"></span></div></div>
  </div>
  <p class="baseline">Class exploration stays at 20%. Task discovery stays at two reserved slots per ten candidates. The slider changes which class receives those discovery slots.</p>
  <p id="forecast-assumption">Illustrative population calculation, not a training simulation or a fitted model. Both classes have unseen tasks. Coverage leaves arithmetic a 10% discovery share even at zero predicted contrast in this two-class example.</p>
</section>
<script>
(() => {
  const el = id => document.getElementById(id);
  function render() {
    const solvedPercent = Number(el('solved-fraction').value);
    const solved = solvedPercent / 100;
    const mixed = (100 - solvedPercent) * 0.875 / 100;
    const algebra = 0.8 * 0.875;
    const share = 0.2 * 0.5 + 0.8 * mixed / (mixed + algebra);
    el('solved-value').textContent = (solved * 100).toFixed(0) + '%';
    el('mixed-value').textContent = (mixed * 100).toFixed(1) + '%';
    el('discovery-value').textContent = (share * 100).toFixed(1) + '%';
    el('mixed-bar').style.width = (mixed * 100) + '%';
    el('discovery-bar').style.width = (share * 100) + '%';
  }
  el('solved-fraction').addEventListener('input', render);
  render();
})();
</script>
"""


def main() -> None:
    original = (ROOT / f"{STEM}.html").read_text()
    head = original[: original.index("</head>")]
    head = re.sub(r"<title>.*?</title>", "<title>Which problem should the student practise next? · v2</title>", head)
    head += f"<style>{CSS}</style></head>"
    html = pypandoc.convert_text(
        (ROOT / f"{STEM}-v2.md").read_text(),
        "html5",
        format="markdown+tex_math_dollars",
        extra_args=["--mathml", "--wrap=none", "--syntax-highlighting=none"],
    )
    reference = BeautifulSoup(html, "html.parser")
    # Keep the formal proposal intact while giving the report a separate narrative.
    reference.h1.name = "h3"
    reader_html = pypandoc.convert_text(
        (ROOT / f"{STEM}-v2-reader.md").read_text(),
        "html5",
        format="markdown+tex_math_dollars",
        extra_args=["--mathml", "--wrap=none", "--syntax-highlighting=none"],
    )
    doc = BeautifulSoup(reader_html, "html.parser")
    reference_panel = doc.new_tag("details", attrs={"class": "reader-reference"})
    summary = doc.new_tag("summary")
    summary.string = "Open the complete v2 technical specification"
    reference_panel.append(summary)
    for element in list(reference.contents):
        reference_panel.append(element.extract())
    doc.find(id="reader-reference").replace_with(reference_panel)
    missing_math = [span.get_text() for span in doc.select(".math") if span.find("math") is None]
    if missing_math:
        raise ValueError(f"Unrendered math: {missing_math}")
    for heading in doc.find_all(re.compile("^h[1-6]$")):
        heading["id"] = re.sub(r"[^a-z0-9]+", "-", heading.get_text().lower()).strip("-")
    for table in doc.find_all("table"):
        wrapper = doc.new_tag("div", attrs={"class": "table-wrap"})
        if table.find_parent("section", class_="calculation-example"):
            wrapper["tabindex"] = "0"
            wrapper["role"] = "region"
            wrapper["aria-label"] = table.caption.get_text() + "; scroll horizontally on narrow screens"
        table.wrap(wrapper)
    for math in doc.select('math[display="block"]'):
        # MathML table cells otherwise shrink fractions to inline/text style.
        # Pandoc also drops TeX row-spacing hints, so CSS supplies row padding.
        math["displaystyle"] = "true"
        for cell in math.find_all("mtd"):
            cell["displaystyle"] = "true"
        wrapper = doc.new_tag("div", attrs={"class": "math-display"})
        parent = math.parent
        if parent is not None and parent.name == "p":
            math.extract()
            wrapper.append(math)
            parent.replace_with(wrapper)
        else:
            math.wrap(wrapper)
    for image in list(doc.find_all("img")):
        source = str(image.get("src", ""))
        if source.endswith(".svg"):
            svg_text = (ROOT / source).read_text()
            for identifier in re.findall(r'\bid="([^"]+)"', svg_text):
                unique = f"{Path(source).stem}-{identifier}"
                svg_text = svg_text.replace(f'id="{identifier}"', f'id="{unique}"')
                svg_text = svg_text.replace(f"url(#{identifier})", f"url(#{unique})")
                svg_text = svg_text.replace(f'href="#{identifier}"', f'href="#{unique}"')
            svg = BeautifulSoup(svg_text, "html.parser").find("svg")
            assert svg is not None
            svg["role"] = "img"
            svg["aria-label"] = str(image.get("alt", "Diagram"))
            wrapper = doc.new_tag("div", attrs={"class": "figure"})
            wrapper.append(svg)
            # Pandoc places the image in a paragraph, with or without a figure.
            parent = image.parent
            assert parent is not None
            if parent.name == "p":
                parent.replace_with(wrapper)
            else:
                image.replace_with(wrapper)
    architecture_pre = next(
        block for block in doc.find_all("pre") if "Task inventory: classes, stable task IDs" in block.get_text()
    )
    architecture_pre.replace_with(BeautifulSoup(ARCHITECTURE, "html.parser"))
    doc.find(id="reader-forecast").replace_with(BeautifulSoup(FORECAST, "html.parser"))
    sim_root = ROOT / "simulations"
    sim_html = (sim_root / "task-discovery-player.html").read_text()
    sim_css = (sim_root / "task-discovery-player.css").read_text()
    sim_js = (sim_root / "task-discovery-engine.js").read_text() + "\n" + (sim_root / "task-discovery-player.js").read_text()
    embedded_sim = f"<style>{sim_css}</style>{sim_html}<script>{sim_js}</script>"
    doc.find(id="reader-simulation").replace_with(BeautifulSoup(embedded_sim, "html.parser"))
    standalone = (
        head.replace("Which problem should the student practise next? · v2", "Task discovery simulation · v2")
        + '<body><main style="max-width:1100px;margin:auto;padding:20px">'
        + embedded_sim
        + "</main></body></html>"
    )
    (sim_root / "task-discovery-v2.html").write_text(standalone)
    nav = BeautifulSoup('<nav aria-label="Contents"><strong>RESEARCH PROPOSAL · VERSION 2</strong></nav>', "html.parser")
    for heading in doc.find_all("h2", recursive=False):
        link = nav.new_tag("a", href="#" + str(heading["id"]))
        link.string = heading.get_text()
        nav.nav.append(link)
    note = '<p class="reader-note">Version 2 reading report, revised 14 September 2026. Fonts, diagrams, both interactive illustrations, and the complete technical specification are embedded. Historical simulations and repository sources open through their linked files.</p>'
    # Old deep links must reveal their target even inside the folded reference.
    reveal = """<script>
function revealReferenceTarget() {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target) return;
  for (let parent = target; parent; parent = parent.parentElement) {
    if (parent.tagName === 'DETAILS') parent.open = true;
  }
  requestAnimationFrame(() => target.scrollIntoView());
}
addEventListener('hashchange', revealReferenceTarget);
revealReferenceTarget();
let printPanels = [];
addEventListener('beforeprint', () => {
  printPanels = [...document.querySelectorAll('details')].filter(panel => !panel.open);
  printPanels.forEach(panel => { panel.open = true; });
});
addEventListener('afterprint', () => printPanels.forEach(panel => { panel.open = false; }));
</script>"""
    result = head + '<body><div class="layout">' + str(nav) + "<article>" + str(doc) + note + "</article></div>" + reveal + "</body></html>"
    output = ROOT / f"{STEM}-v2.html"
    output.write_text(result)
    print(f"Rendered {output.name}: {len(doc.find_all('h2'))} sections, {len(doc.find_all('math'))} equations and inline expressions.")
    portable = BeautifulSoup(result, "html.parser")
    portable.select_one(".reader-demo")["id"] = "embedded-task-simulation"
    attachments = {}
    for link in portable.select("a[href]"):
        href = str(link["href"])
        parsed = urlsplit(href)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        if parsed.path == "simulations/task-discovery-v2.html":
            link["href"] = "#embedded-task-simulation"
            continue
        path = (ROOT / unquote(parsed.path)).resolve()
        if not path.is_file():
            raise ValueError(f"Missing standalone attachment: {href}")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        attachments[path.name] = (mime, base64.b64encode(path.read_bytes()).decode("ascii"))
        link["href"] = f"data:{mime};base64,{attachments[path.name][1]}"
        link["download"] = path.name
        link["title"] = f"Download embedded reference: {path.name}"
        link.append(" (download)")
    portable.select_one(".reader-note").string = (
        "Standalone reading report. Open this HTML directly in a browser; no server or installation is needed. "
        "Fonts, diagrams, equations, examples, simulation, and the technical specification are embedded. "
        "Repository references download from this file. Links inside downloaded historical references retain "
        "their original destinations. External research links require internet access."
    )
    for element in portable.select("[src], link[href]"):
        source = str(element.get("src") or element.get("href"))
        if not source.startswith(("data:", "#")):
            raise ValueError(f"Standalone report has an external asset: {source}")
    portable_output = ROOT / f"{STEM}-v2-standalone.html"
    portable_output.write_text(str(portable))
    print(f"Rendered {portable_output.name}: {portable_output.stat().st_size:,} bytes, {len(attachments)} embedded reference files.")


if __name__ == "__main__":
    main()
