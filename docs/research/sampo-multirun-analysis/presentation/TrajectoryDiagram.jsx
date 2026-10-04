import React, {useState} from "react";

const number=v=>`${v>0?"+":""}${v.toFixed(5)}`;
const names={google_sheets_get_spreadsheet_by_id:"Find sheet",google_sheets_get_many_rows:"Read rows",gmail_send_email:"Send email",google_sheets_update_row:"Update status",gmail_find_email:"Search Gmail",google_ads_find_campaigns:"Find campaigns",google_ads_find_campaign_by_name:"Find campaign",google_ads_set_campaign_status:"Pause campaign",google_drive_find_multiple_files:"Search Drive"};
function Blocks({value,masked=false}) {
 return <div className={`trajectory-blocks ${value<0?"negative":"positive"} ${masked?"masked":""}`} aria-label={`Sampled token spans, credit ${number(value)}`}>
  {Array.from({length:8},(_,i)=><span key={i}>{value<0?"−":"+"}</span>)}
 </div>;
}
function Turn({turn,action,focus=false}) {
 const calls=turn.calls.map(c=>names[c.name]??c.name);
 return <div className={`trajectory-turn ${focus?"focus":""}`}>
  <header><b>T{turn.turn}</b><span>{turn.token_count.toLocaleString()} tokens</span></header>
  <div className="trajectory-action" title={turn.calls.map(c=>c.name).join(" · ")}>{action??([...new Set(calls)].join(" + ")||"Final response")}</div>
  <Blocks value={turn.combined_advantage}/>
  <div className="trajectory-value">A {number(turn.combined_advantage)}</div>
  {focus&&<div className="trajectory-event">Δ score {number(turn.immediate_assertion_progress)} · anchor {turn.anchor_size}</div>}
 </div>;
}
export function TurnEvidence({turn,credit=turn.combined_advantage,mask=false,compact=false}) {
 const brief=(s,n)=>compact&&s.length>n?s.slice(0,n).trim()+"…":s;
 return <div className={`turn-evidence ${mask?"cost-masked":""}`}>
  <header><b>Turn {turn.turn}</b><span>{turn.token_count.toLocaleString()} sampled tokens</span></header>
  <dl>
   <dt>Thinking</dt><dd><blockquote>{brief(turn.thinking_excerpt,170)||"No separate thinking text was recorded."}</blockquote></dd>
   <dt>Action</dt><dd>{turn.calls.length?turn.calls.map((c,i)=><code key={i}>{c.name}({brief(c.arguments_excerpt,120)})</code>):<code>Final answer: {turn.answer_excerpt}</code>}</dd>
   <dt>Result</dt><dd>{turn.result_excerpts.length?turn.result_excerpts.map((r,i)=><code key={i}>{brief(r,140)}</code>):<span>End of episode</span>}</dd>
  </dl>
  <div className="turn-credit-support"><span>{mask?"Proposed cost mask · entire sampled turn":"Current-recipe credit · entire sampled turn"}</span><Blocks value={credit} masked={mask}/><strong>Credit {number(credit)}</strong></div>
  <small>Thinking is a shortened native excerpt; calls and results are abridged. Tool results are observations, excluded from the policy loss mask.</small>
 </div>;
}
function RecordedTurns({trace,kind}) {
 const [selected,setSelected]=useState(kind==="vendor"?3:8);
 const turn=trace.turns[selected];
 return <><div className="recorded-turn-selector" aria-label="Recorded assistant turns">{trace.turns.map(t=><button type="button" aria-pressed={selected===t.turn} onClick={()=>setSelected(t.turn)} key={t.turn}><b>Turn {t.turn}</b><span>{[...new Set(t.calls.map(c=>names[c.name]??c.name))].join(" + ")||"Final answer"}</span></button>)}</div><TurnEvidence turn={turn}/><div className="turn-replay-reading"><span>Episode credit <b>{number(trace.episode_credit)}</b></span><span>Local comparison <b>{turn.anchor_size===1?"None · singleton":"Anchor size "+turn.anchor_size}</b></span><span>Task-score change <b>{number(turn.immediate_assertion_progress)}</b></span></div></>;
}
export function TrajectoryDiagram({kind,views}) {
 const [method,setMethod]=useState("replace");
 const vendor=views.vendor;
 const ads=views.ads;
 const bad=vendor.turns[3];
 const cost=Math.max(0,-bad.immediate_native_turn_reward);
 if(kind==="mask") return <figure id="mask-comparison" className="trajectory-figure" aria-label="Credit-mask comparison on the recorded forbidden email">
  <figcaption><a href="#mask-comparison">Same forbidden email · T3 · {bad.token_count} sampled tokens</a></figcaption>
  <div className="mask-method-control" aria-label="Counterfactual cost method"><button type="button" aria-pressed={method==="subtract"} onClick={()=>setMethod("subtract")}>Subtract cost</button><button type="button" aria-pressed={method==="replace"} onClick={()=>setMethod("replace")}>Replace credit on mask</button></div>
  <div className="real-mask-comparison"><section><h4>Current-recipe replay</h4><TurnEvidence turn={bad} compact/></section><section><h4>{method==="subtract"?"Subtract cost · κ = 1":"Replace on cost mask · κ = 1"}</h4><TurnEvidence turn={bad} credit={method==="subtract"?bad.combined_advantage-cost:-cost} mask compact/></section></div>
  <p className="mask-explanation">{method==="subtract"?"The observed cost reduces credit, but this turn still has a positive label. Subtraction does not guarantee a harmful turn is discouraged.":"The cost mask replaces positive task credit with negative cost credit on this turn. Other turns keep their original labels."}</p>
  <div className="trajectory-legend"><span className="trajectory-mask-key"/> Proposed whole-turn cost mask <span>Other turns retain their credit · tool outputs excluded</span></div>
  <small>Counterfactual credit only; no model update. Cost = −native turn reward. Exact call-token boundaries are unavailable.</small>
 </figure>;
 const trace=kind==="vendor"?vendor:ads;
 return <figure className="trajectory-figure" aria-label={trace.label}>
  <figcaption>{trace.label}{kind==="ads"&&<span className="trajectory-success">Complete success</span>}</figcaption>
  <RecordedTurns trace={trace} kind={kind}/>
  <div className="trajectory-legend"><span>+ positive credit</span><span>− negative credit</span><span>Eight blocks summarize the selected turn's native policy mask; block widths are not token counts.</span></div>
  <small>{kind==="vendor"?"Replay: mean centering, turn weight 1. T3 has zero local turn credit; its positive episode credit survives.":"Recorded successful branch: Gmail retrieval at T3; four campaign writes at T8. Five siblings failed at the same checkpoint."}</small>
  <details><summary>Trace and replay source</summary><p>Native episode: {trace.native_episode_id}<br/>Source SHA-256: {trace.native_sha256}<br/>Current-code replay SHA-256: {trace.replay_sha256}</p></details>
 </figure>;
}
