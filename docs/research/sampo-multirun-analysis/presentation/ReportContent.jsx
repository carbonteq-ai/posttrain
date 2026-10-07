import React, { useState } from "react";
import { TrajectoryDiagram } from "./TrajectoryDiagram.jsx";
import { StrategyVisual } from "./StrategyVisual.jsx";
import { DataComponent, DataTable, EvidenceChart, ReportSection, RichNarrative, useDataApp } from "../../data-app-public.jsx";
const pct = v => v == null ? "Unavailable" : `${(100*v).toFixed(1)}%`;
const decimal = v => v == null ? "Unavailable" : v.toFixed(3);
const columns = [{field:"label",label:"Policy / checkpoint"},{field:"temperature",label:"Temperature"},{field:"episodes",label:"Episodes"},{field:"taskReward",label:"Task reward",renderCell:decimal},{field:"nonSimpleReward",label:"Non-Simple",renderCell:decimal},{field:"guardViolation",label:"Guard violations",renderCell:pct}];
const groupColumns = [{field:"label",label:"Training run"},{field:"episodes",label:"Episodes"},{field:"groups",label:"Complete groups"},{field:"simpleZeroVariance",label:"Simple: constant reward",renderCell:pct},{field:"hardZeroVariance",label:"Non-Simple: constant reward",renderCell:pct},{field:"neverPass",label:"Assertions never passed",renderCell:pct}];
const trainingColumns = [{field:"label",label:"Training arm"},{field:"episodes",label:"Stored episodes"},{field:"early",label:"Early hard-task reward",renderCell:decimal},{field:"late",label:"Late hard-task reward",renderCell:decimal},{field:"matchedTasks",label:"Matched tasks"},{field:"truncation",label:"Truncated",renderCell:pct}];
function NarrativeBlocks({section}) {
 const blocks=section.body.split(/\n\s*\n/);
 return blocks.map((block,index)=>{
 const sourceIndex=(section.originalBody??section.body).split(/\n\s*\n/).findIndex(b=>b.trim()===block.trim());
 const narrativeId=sourceIndex<0?`${section.id}:reading:${index}`:`${section.id}:body:${sourceIndex}`;
 const lines=block.trim().split("\n");
  const trajectory=block.trim().match(/^\[\[trajectory:([a-z]+)\]\]$/);
  if(trajectory) return <TrajectoryDiagram key={index} kind={trajectory[1]} views={section.trajectoryViews}/>;
  if (block.trim().startsWith("$$") && block.trim().endsWith("$$")) {
   const latex=block.trim().slice(2,-2).trim();
   const math=section.equationMath?.[latex];
   if (!math) throw new Error(`Missing rendered LaTeX in ${section.id}`);
   return <figure className="sampo-equation" key={index} aria-label="Report equation" title={latex}><div dangerouslySetInnerHTML={{__html:math}}/></figure>;
  }
  if(lines.length>1&&lines[0].startsWith("|")&&/^\|[\s:|\-]+\|$/.test(lines[1])){
   const cells=line=>line.replace(/^\||\|$/g,"").split("|").map(c=>c.trim());
   const headers=cells(lines[0]);
   return <div className="sampo-narrative-table" key={index}><table aria-label={`${section.title} evidence table ${index}`}><thead><tr>{headers.map((h,j)=><th key={j} scope="col">{h}</th>)}</tr></thead><tbody>{lines.slice(2).map((line,i)=><tr key={i}>{cells(line).map((v,j)=><td key={j}>{v.replace(/`/g,"")}</td>)}</tr>)}</tbody></table></div>;
  }
  return <RichNarrative key={index} id={narrativeId} value={block}/>;
 });
}
const chapters={
 "report-summary":["01", "The finding", "Parts of the workflow improve; complete procedures remain unreliable."],
 "report-hypothesis":["02", "Why learning may stall", "The evidence supports several interacting mechanisms. Their individual effects still need a controlled run."],
 "report-training":["03", "Across actual training runs", "Compare the same hard tasks before interpreting averages from a changing task mix."],
 "report-simulations":["04", "What each strategy changes", "These experiments replay recorded outcomes and credit. They do not measure a newly trained policy."],
 "report-next":["05", "The next training comparison", "Establish the corrected loss first, then test local costs and practice separately."]
};
function ReadingSection({section}) {
 section={...section,originalBody:section.body};
 const chunks=section.body.split(/\n\s*\n/);
 if(section.id==="report-summary") return <><NarrativeBlocks section={{...section,body:chunks[1]}}/><details className="sampo-disclosure"><summary>Diagnosis, scope and limits</summary><NarrativeBlocks section={{...section,body:chunks.slice(2).join("\n\n")}}/></details></>;
 if(section.id==="report-hypothesis"||section.id==="report-training") {
  const end=section.id==="report-training"?2:3;
  return <><NarrativeBlocks section={{...section,body:chunks.slice(1,end).join("\n\n")}}/><details className="sampo-disclosure"><summary>{section.id==="report-training"?"Task-level trajectories, exact tables and measurement limits":"Mechanisms, competing hypotheses and tests"}</summary><NarrativeBlocks section={{...section,body:chunks.slice(end).join("\n\n")}}/></details></>;
 }
 if(section.id==="report-simulations"||section.id==="report-next") {
  const parts=section.body.split(/(?=^### )/m);
  return <><NarrativeBlocks section={{...section,body:parts[0].replace(/^## .*\n/gm,"")}}/>{parts.slice(1).map((p,i)=>{
   const title=p.split("\n")[0].replace(/^### /,"");
   const body=p.substring(p.indexOf("\n")+1);
   const preview=body.trim().split(/\n\s*\n/)[0].split(/\.\s/)[0]+".";
   return <details className="sampo-strategy" key={title} open={section.id==="report-next"?i===0:i===3}><summary><span className="sampo-strategy-index">{String(i+1).padStart(2,"0")}</span>{title}<small className="sampo-strategy-preview">{preview}</small></summary>{section.id==="report-simulations"&&<StrategyVisual title={title} views={section.trajectoryViews} body={body}/>}<NarrativeBlocks section={{...section,body:body.replace("[[trajectory:vendor]]","")}}/></details>;
  })}</>;
 }
 return <NarrativeBlocks section={{...section,body:section.body.replace(/^## .*\n/,"")}}/>;
}
export function ReportContent(){
 const {snapshot,reviewedRows,visible,canEdit,mode,appTitle,setAppTitle}=useDataApp();
 const [temperature,setTemperature]=useState("all");
 const all=reviewedRows("checkpoint_results");
 const evals=all.filter(r=>temperature==="all"||String(r.temperature)===temperature);
 const groups=snapshot.queries.training_groups ? reviewedRows("training_groups") : [];
 const checkpoints=snapshot.queries.checkpoint_curve ? reviewedRows("checkpoint_curve") : [];
 const training=snapshot.queries.training_results ? reviewedRows("training_results") : [];
 const trainingCurve=snapshot.queries.training_curve ? reviewedRows("training_curve") : [];
 const vendorViews=(snapshot.report?.sections??[]).find(s=>s.trajectoryViews)?.trajectoryViews;
 return <article className="report-content" aria-label="SAMPO evidence report">
  <header className="report-hero"><div className="sampo-eyebrow">SAMPO · Training evidence and system diagnosis</div><h1 data-data-app-title contentEditable={canEdit&&mode==="edit"} suppressContentEditableWarning onBlur={canEdit&&mode==="edit" ? e=>setAppTitle(e.currentTarget.textContent.trim()||appTitle):undefined}>{appTitle}</h1><p className="sampo-deck">Why partial progress does not become reliable conditional behavior, and what to test next.</p><p className="sampo-scope">74,006 stored training episodes · 7 substantial runs + 1 interrupted attempt · 14 evaluation datasets</p></header>
  <nav className="sampo-reading-nav" aria-label="Report reading path"><a href="#report-summary">Finding</a><a href="#recorded-action">Real trajectory</a><a href="#report-hypothesis">Diagnosis</a><a href="#report-training">Training runs</a><a href="#report-simulations">Strategies</a><a href="#report-next">Pilot</a><a href="#report-methods">Sources</a></nav>
  {(snapshot.report?.sections??[]).map(s=><React.Fragment key={s.id}>
   {visible(s.id)&&<section id={s.id} className={`sampo-chapter ${chapters[s.id]?"major":"supporting"}`}>
    {chapters[s.id]&&<header className="sampo-chapter-heading"><span>{chapters[s.id][0]}</span><div><h2>{chapters[s.id][1]}</h2><p>{chapters[s.id][2]}</p></div></header>}
    {s.id==="report-methods"||s.id==="report-claims"||s.id==="report-evaluations"?<details className="sampo-disclosure"><summary>{s.title}</summary><ReportSection id={s.id} title={s.title} queryId={s.queryId} queryIds={s.queryIds??[s.queryId]} sourceRowsByQuery={Object.fromEntries((s.queryIds??[s.queryId]).map(q=>[q,reviewedRows(q)]))} showHeading={false}><ReadingSection section={s}/></ReportSection></details>:<ReportSection id={s.id} title={s.title} queryId={s.queryId} queryIds={s.queryIds??[s.queryId]} sourceRowsByQuery={Object.fromEntries((s.queryIds??[s.queryId]).map(q=>[q,reviewedRows(q)]))} showHeading={!chapters[s.id]}><ReadingSection section={s}/></ReportSection>}
   </section>}
   {s.id==="report-summary"&&vendorViews&&<section id="recorded-action" className="sampo-case-study"><div className="sampo-eyebrow">Recorded action · Reconstructed credit</div><h2>A prohibited email can still receive positive credit</h2><p>The model sent the email in a real training rollout. Replaying the current recipe gives that turn positive episode credit and zero relative turn credit because its anchor has no sibling comparison.</p><TrajectoryDiagram kind="vendor" views={vendorViews}/><p className="sampo-case-limit">This establishes a credit-sign conflict in the replay. It does not establish how much that conflict caused the historical training failure.</p></section>}
   {s.id==="report-evaluations"&&visible("checkpoint-results")&&<DataComponent id="checkpoint-results" title="Checkpoint results on the same held-out tasks" queryId="checkpoint_results" kind="table" sourceRows={evals} displayRows={evals}>
    <div className="sampo-control"><label htmlFor="temperature">Sampling temperature</label><select id="temperature" value={temperature} onChange={e=>setTemperature(e.target.value)}><option value="all">Both temperatures</option><option value="0.1">0.1 · 3 attempts per task</option><option value="0.5">0.5 · 5 attempts per task</option></select></div>
    <DataTable rows={evals} columns={columns} label="SAMPO checkpoint comparison" searchable={false}/>
   </DataComponent>}
   {s.id==="report-simple"&&groups.length>0&&<DataComponent id="training-groups" title="Simple tasks leave little reward variation; hard tasks still vary" queryId="training_groups" sourceRows={groups} displayRows={groups} kind="table"><DataTable rows={groups} columns={groupColumns} searchable={false} label="Training group comparison"/></DataComponent>}
   {s.id==="report-training"&&training.length>0&&<DataComponent id="training-results" title="Actual training: same hard tasks in early and late windows" queryId="training_results" sourceRows={training} displayRows={training} kind="table"><DataTable rows={training} columns={trainingColumns} searchable={false} label="Actual training comparison"/></DataComponent>}
   {s.id==="report-training"&&trainingCurve.length>0&&<EvidenceChart id="training-curve" title="Training candidate reward · ten-step means · changing task mix" queryId="training_curve" rows={trainingCurve} sourceRows={trainingCurve} height={360} spec={{type:"line",x:"step",y:"taskReward",series:"label",stackable:false,valueDecimals:3}}/>}
   {s.id==="report-evaluations"&&checkpoints.length>0&&<EvidenceChart id="checkpoint-curve" title="Held-out non-Simple task reward by checkpoint · temperature 0.5" queryId="checkpoint_curve" rows={checkpoints} sourceRows={checkpoints} height={340} spec={{type:"line",x:"checkpointStep",y:"taskReward",series:"label",stackable:false,valueDecimals:3}}/>}
  </React.Fragment>)}
 </article>;
}
