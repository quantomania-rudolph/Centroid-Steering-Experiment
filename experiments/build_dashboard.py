#!/usr/bin/env python
"""Build J-space workspace dashboard from experiment JSON runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "out"
DASH_DIR = ROOT / "experiments" / "dashboard"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>J-Space Workspace Gallery</title>
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600&family=JetBrains+Mono:wght@400;500&family=Sora:wght@300;400;500;600&display=swap" rel="stylesheet" />
<style>
:root {
  --emerald-deep:#041a14; --emerald:#0b3d2e; --emerald-mid:#146b4f; --emerald-glow:#1fa87a;
  --silver:#d7dde3; --silver-dim:#9aa6b2; --gold:#d4af37; --gold-soft:#f0d78c;
  --ink:#eef3f0; --panel:rgba(8,42,32,0.88); --line:rgba(212,175,55,0.45);
  --line-silver:rgba(215,221,227,0.28); --pass:#6ec9a0; --fail:#f0a060;
  --workspace:rgba(31,168,122,0.25);
}
*{box-sizing:border-box}
html,body{margin:0;min-height:100%;color:var(--ink);font-family:"Sora",system-ui,sans-serif;
  background:radial-gradient(1200px 700px at 12% -10%,rgba(212,175,55,0.18),transparent 55%),
  radial-gradient(900px 600px at 110% 10%,rgba(31,168,122,0.22),transparent 50%),
  linear-gradient(165deg,#020c09 0%,var(--emerald-deep) 38%,#062820 72%,#031510 100%)}
.app{max-width:1500px;margin:0 auto;padding:20px 16px 64px}
.hero{border:1px solid var(--line);padding:22px 24px;margin-bottom:16px;
  background:linear-gradient(135deg,rgba(212,175,55,0.12),transparent 40%),linear-gradient(180deg,rgba(20,107,79,0.35),rgba(4,26,20,0.55))}
.brand{font-family:"Cormorant Garamond",serif;font-size:2.4rem;font-weight:600;
  background:linear-gradient(100deg,var(--gold-soft),var(--silver),var(--gold));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.tagline{color:var(--silver-dim);font-size:0.9rem;margin-top:6px;line-height:1.5}
.status-bar{margin-top:10px;padding:8px 12px;border:1px solid var(--line-silver);
  background:rgba(0,0,0,0.25);font-size:0.78rem;color:var(--gold-soft)}
.layout{display:grid;grid-template-columns:240px 1fr;gap:14px}
@media(max-width:900px){.layout{grid-template-columns:1fr}}
.panel{border:1px solid var(--line-silver);background:var(--panel);backdrop-filter:blur(10px)}
.panel h2{margin:0;padding:11px 14px;font-size:0.66rem;letter-spacing:0.14em;
  text-transform:uppercase;color:var(--gold-soft);border-bottom:1px solid var(--line-silver)}
.run-btn{display:block;width:100%;text-align:left;border:1px solid transparent;background:transparent;
  color:var(--ink);padding:10px 12px;margin-bottom:5px;cursor:pointer}
.run-btn:hover{border-color:var(--line-silver);background:rgba(31,168,122,0.12)}
.run-btn.active{border-color:var(--gold);background:linear-gradient(90deg,rgba(212,175,55,0.16),rgba(31,168,122,0.12))}
.run-btn .name{display:block;font-family:"Cormorant Garamond",serif;font-size:1.05rem;color:var(--silver)}
.run-btn .meta{font-size:0.68rem;color:var(--silver-dim);margin-top:2px}
.main{padding:14px}
.stats{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:12px}
.stat{border:1px solid var(--line-silver);padding:6px 10px;background:rgba(4,26,20,0.45);min-width:90px}
.stat b{display:block;font-size:0.6rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--gold)}
.stat span{font-size:0.82rem;color:var(--silver)}
.section{margin:14px 0 8px;font-size:0.66rem;letter-spacing:0.12em;text-transform:uppercase;color:var(--gold)}
.workspace-chart{display:flex;align-items:flex-end;gap:3px;height:120px;padding:10px;
  border:1px solid var(--line);background:rgba(0,0,0,0.2);margin-bottom:14px}
.wbar{flex:1;min-width:8px;background:linear-gradient(180deg,var(--emerald-glow),var(--emerald-mid));
  border:1px solid rgba(212,175,55,0.3);position:relative;cursor:pointer;transition:opacity 0.15s}
.wbar.workspace{background:linear-gradient(180deg,var(--gold-soft),var(--gold))}
.wbar:hover{opacity:0.85}
.wbar .lbl{position:absolute;bottom:-18px;left:50%;transform:translateX(-50%);
  font-size:0.58rem;color:var(--silver-dim);font-family:"JetBrains Mono",monospace}
.evidence{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
@media(max-width:700px){.evidence{grid-template-columns:1fr}}
.ev{display:flex;gap:8px;align-items:flex-start;padding:8px 10px;border:1px solid var(--line-silver);
  background:rgba(0,0,0,0.2);font-size:0.76rem}
.ev.pass{border-color:rgba(110,201,160,0.5)} .ev.fail{border-color:rgba(240,160,96,0.5)}
.ev .mark{font-weight:600;font-size:0.65rem;letter-spacing:0.08em}
.ev.pass .mark{color:var(--pass)} .ev.fail .mark{color:var(--fail)}
.layer-strip{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:12px}
.layer-chip{border:1px solid var(--line-silver);background:rgba(4,26,20,0.5);color:var(--silver);
  padding:5px 9px;font-family:"JetBrains Mono",monospace;font-size:0.72rem;cursor:pointer}
.layer-chip.workspace{border-color:var(--gold);color:var(--gold-soft)}
.layer-chip.active{border-color:var(--gold);background:rgba(212,175,55,0.2);color:var(--gold-soft)}
.inv-strip{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px}
.inv-chip{border:1px solid var(--line-silver);background:rgba(4,26,20,0.55);color:var(--silver);
  padding:7px 10px;font-family:"JetBrains Mono",monospace;font-size:0.7rem;cursor:pointer;text-align:left}
.inv-chip.active{border-color:var(--gold);background:rgba(212,175,55,0.18);color:var(--gold-soft)}
.inv-chip .sub{display:block;color:var(--silver-dim);font-size:0.62rem;margin-top:2px}
.assistant-box{max-height:160px;overflow:auto;border:1px solid var(--line-silver);padding:8px;
  font-family:"JetBrains Mono",monospace;font-size:0.72rem;color:var(--silver);white-space:pre-wrap;margin-bottom:12px}
.rep-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:800px){.rep-grid{grid-template-columns:1fr}}
.rep-box{border:1px solid var(--line-silver);padding:10px;background:rgba(2,12,9,0.35)}
.rep-box h3{margin:0 0 8px;font-size:0.64rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--emerald-glow)}
.rep-row{display:grid;grid-template-columns:1fr 60px;gap:6px;align-items:center;margin-bottom:6px}
.rep-token{font-family:"JetBrains Mono",monospace;font-size:0.78rem;padding:4px 7px;
  border:1px solid rgba(212,175,55,0.3);background:rgba(31,168,122,0.1);color:var(--gold-soft);
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bar{height:6px;background:rgba(215,221,227,0.12);overflow:hidden}
.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--emerald-mid),var(--gold))}
.traj-wrap{overflow:auto;max-height:200px;border:1px solid var(--line);padding:8px;
  font-family:"JetBrains Mono",monospace;font-size:0.72rem;line-height:1.6;color:var(--silver)}
.traj-wrap .pos{color:var(--gold);margin-right:4px}
.sparse{margin-top:10px;padding:8px;border:1px solid var(--line-silver);font-size:0.74rem}
.sparse .atom{display:inline-block;margin:3px;padding:3px 7px;border:1px solid var(--line-silver);
  background:rgba(31,168,122,0.08);font-family:"JetBrains Mono",monospace}
.empty{padding:30px;text-align:center;color:var(--silver-dim)}
.chaos-radar{margin:10px 0;padding:10px;border:1px solid var(--line-silver);border-radius:8px;
  background:rgba(0,0,0,0.25)}
.chaos-badges{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
.chaos-badge{padding:4px 10px;border:1px solid var(--line-silver);border-radius:999px;font-size:0.72rem}
.chaos-badge.warn{border-color:#c9a227;color:#f0d78c}
.chaos-badge.bad{border-color:#c45c5c;color:#f0a0a0}
.flag-timeline{display:flex;flex-wrap:wrap;gap:4px;margin:6px 0 10px}
.flag-chip{font-size:0.65rem;padding:2px 6px;border:1px solid #c45c5c;color:#f0a0a0;border-radius:4px}
.chaos-heat{display:grid;gap:2px;margin-top:6px}
.chaos-heat .row{display:flex;align-items:center;gap:4px}
.chaos-heat .ylab{width:36px;font-size:0.65rem;color:var(--silver-dim)}
.chaos-heat .cell{width:10px;height:14px;border-radius:2px;background:rgba(31,168,122,0.15)}
.ag-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin:10px 0}
.ag-card{border:1px solid var(--line-silver);padding:8px 10px;font-size:0.72rem;background:rgba(0,0,0,0.2)}
.ag-card.ok{border-color:rgba(110,201,160,0.45)}
.ag-card.warn{border-color:rgba(201,162,39,0.55)}
.ag-card.fail{border-color:rgba(196,92,92,0.55)}
.ag-card.info{border-color:rgba(154,166,178,0.35)}
.ag-card .id{font-family:"JetBrains Mono",monospace;color:var(--gold-soft);font-size:0.78rem}
.ag-card .st{margin-top:4px;color:var(--silver-dim);font-size:0.65rem;line-height:1.35}
</style>
</head>
<body>
<div class="app">
  <header class="hero">
    <div class="brand">J-Space Workspace</div>
    <p class="tagline">Jacobian-lens workspace readouts — top-20 representations per layer, trajectory across tokens,
      sparse decomposition, and evidence that J-space captures model internals.</p>
    <div class="status-bar" id="statusBar">Loading…</div>
  </header>
  <div class="layout">
    <aside class="panel">
      <h2>Experiments</h2>
      <div id="runList" style="padding:8px"></div>
    </aside>
    <section class="panel main" id="mainPanel">
      <div class="empty" id="empty">Select an experiment.</div>
      <div id="detail" style="display:none">
        <div class="stats" id="stats"></div>
        <div class="section">Workspace boundary (topk_acc per layer)</div>
        <div class="workspace-chart" id="workspaceChart"></div>
        <div class="section">Evidence checks</div>
        <div class="evidence" id="evidenceBox"></div>
        <div class="section" id="invSection" style="display:none">Agentic invocations (LLM calls)</div>
        <div class="inv-strip" id="invStrip" style="display:none"></div>
        <div id="assistantPreview" class="assistant-box" style="display:none"></div>
        <div class="section" id="chaosSection" style="display:none">Chaos radar (A–G live)</div>
        <div class="chaos-radar" id="chaosRadar" style="display:none"></div>
        <div class="section">Layers — click to inspect J-space top-20</div>
        <div class="layer-strip" id="layerStrip"></div>
        <div class="section" id="layerTitle">Layer detail</div>
        <div class="rep-grid" id="repGrid"></div>
        <div class="section">Top-1 trajectory (J-lens across sequence)</div>
        <div class="traj-wrap" id="trajectory"></div>
        <div id="sparseBox"></div>
      </div>
    </section>
  </div>
</div>
<script>
let RUNS=[], active=null, layerIdx=0, invIdx=0, workspaceBand=null, activeLayers=[];
const el=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function runLabel(r){return r.experiment||r.stamp}
function setStatus(m){el("statusBar").textContent=m}
function currentLayerRows(){
  if(active&&active.kind==="aider_agentic"&&(active.invocations||[]).length){
    const inv=active.invocations[invIdx]||{};
    return (inv.jspace&&inv.jspace.layers)||[];
  }
  return active? (active.layers||[]) : [];
}
function currentKeyPositions(){
  if(active&&active.kind==="aider_agentic"&&(active.invocations||[]).length){
    const inv=active.invocations[invIdx]||{};
    return (inv.jspace&&inv.jspace.key_positions)||{};
  }
  return (active&&active.key_positions)||{};
}

async function loadRuns(){
  try{
    const res=await fetch("data.json",{cache:"no-store"});
    RUNS=await res.json();
    setStatus("Loaded "+RUNS.length+" experiment(s)");
  }catch(e){
    const inline=document.getElementById("runs-inline");
    if(inline){RUNS=JSON.parse(inline.textContent);setStatus("Inline fallback: "+RUNS.length+" runs")}
    else{setStatus("Load failed — run: python experiments/serve_dashboard.py");RUNS=[]}
  }
  renderRunList();
  if(RUNS.length) selectRun(0);
}

function renderRunList(){
  const box=el("runList");
  if(!RUNS.length){box.innerHTML="<p class='empty'>No experiments</p>";return}
  box.innerHTML=RUNS.map((r,i)=>{
    const kind=r.kind||"legacy";
    const ev=r.evidence?`${r.evidence.passed}/${r.evidence.total} pass`:"—";
    const nInv=(r.invocations||[]).length;
    const invMeta=nInv?` · ${nInv} inv` : "";
    return `<button class="run-btn" data-i="${i}"><span class="name">${esc(runLabel(r))}</span>
      <span class="meta">${esc(r.stamp)} · ${kind}${invMeta} · ${ev}</span></button>`;
  }).join("");
  box.querySelectorAll(".run-btn").forEach(b=>b.addEventListener("click",()=>selectRun(+b.dataset.i)));
}

function renderReps(title, tokens, useCoeff){
  if(!tokens||!tokens.length)return "";
  const maxVal=Math.max(...tokens.map(t=>useCoeff?t.coeff||0:t.prob||0),1e-9);
  let h=`<div class="rep-box"><h3>${esc(title)}</h3>`;
  tokens.forEach(t=>{
    const v=useCoeff?(t.coeff||0):(t.prob||0);
    const pct=(v/maxVal*100).toFixed(1);
    const label=useCoeff?`c=${v.toFixed(3)}`:`p=${v.toFixed(3)}`;
    h+=`<div class="rep-row"><div class="rep-token" title="${esc(t.token)}">${esc(t.token)}</div>
      <div class="bar"><i style="width:${pct}%"></i></div></div>
      <div style="font-size:0.62rem;color:var(--silver-dim);margin:-2px 0 6px">${label}</div>`;
  });
  return h+"</div>";
}

function selectRun(i){
  active=RUNS[i]; layerIdx=0; invIdx=0;
  el("empty").style.display="none"; el("detail").style.display="block";
  document.querySelectorAll(".run-btn").forEach((b,bi)=>b.classList.toggle("active",bi===i));

  const ws=active.workspace||{};
  workspaceBand=ws.workspace_band;
  const gen=active.generation||{};
  const nInv=(active.invocations||[]).length;
  el("stats").innerHTML=[
    ["Experiment",runLabel(active)],
    ["Kind",active.kind||"legacy"],
    ["Invocations", nInv||"—"],
    ["Seq len",active.seq_len||"—"],
    ["Workspace",ws.workspace_band?`L${ws.workspace_band[0]}–L${ws.workspace_band[1]}`:"—"],
    ["Device",active.device||"—"],
  ].map(([k,v])=>`<div class="stat"><b>${esc(k)}</b><span>${esc(v)}</span></div>`).join("");

  // workspace chart
  const perLayer=ws.per_layer||{};
  const chartEl=el("workspaceChart");
  const layers=Object.keys(perLayer).map(Number).sort((a,b)=>a-b);
  if(!layers.length){chartEl.innerHTML="<div class='empty' style='padding:8px'>No workspace metrics</div>";}
  else{
    const maxAcc=Math.max(...layers.map(l=>perLayer[String(l)].topk_acc||0),0.01);
    chartEl.innerHTML=layers.map(l=>{
      const acc=perLayer[String(l)].topk_acc||0;
      const h=Math.max(4,acc/maxAcc*100);
      const inBand=workspaceBand&&l>=workspaceBand[0]&&l<=workspaceBand[1];
      return `<div class="wbar${inBand?" workspace":""}" style="height:${h}%"
        title="L${l} acc=${acc.toFixed(3)}"><span class="lbl">L${l}</span></div>`;
    }).join("");
  }

  // evidence
  const evBox=el("evidenceBox");
  const checks=(active.evidence&&active.evidence.checks)||[];
  evBox.innerHTML=checks.map(c=>{
    const cls=c.pass?"pass":"fail";
    return `<div class="ev ${cls}"><span class="mark">${c.pass?"PASS":"FAIL"}</span>
      <div><b>${esc(c.id)}</b> ${esc(c.name)}<br><span style="color:var(--silver-dim)">${esc(c.detail)}</span></div></div>`;
  }).join("")||"<p class='empty'>No evidence data</p>";

  // invocations
  const invs=active.invocations||[];
  const invSection=el("invSection");
  const invStrip=el("invStrip");
  const asst=el("assistantPreview");
  if(invs.length){
    invSection.style.display="block";
    invStrip.style.display="flex";
    asst.style.display="block";
    invStrip.innerHTML=invs.map((inv,idx)=>{
      const nLay=((inv.jspace||{}).layers||[]).length;
      const tok=inv.completion_tokens??"—";
      const hasChaos=!!inv.chaos;
      const plds=inv.chaos&&inv.chaos.composites?Number(inv.chaos.composites.PLDS||0).toFixed(2):"";
      return `<button class="inv-chip" data-idx="${idx}">inv #${inv.invocation_index??idx+1}
        <span class="sub">${tok} tok · ${nLay} layers${hasChaos?` · PLDS ${plds}`:""}</span></button>`;
    }).join("");
    invStrip.querySelectorAll(".inv-chip").forEach(chip=>{
      chip.addEventListener("click",()=>{
        invIdx=+chip.dataset.idx;
        layerIdx=0;
        renderInvocation();
      });
    });
  }else{
    invSection.style.display="none";
    invStrip.style.display="none";
    asst.style.display="none";
  }

  renderInvocation();
  setStatus(`Viewing ${runLabel(active)} · ${nInv||0} inv · workspace ${ws.workspace_band||"?"}`);
}

function renderInvocation(){
  const invs=active.invocations||[];
  const asst=el("assistantPreview");
  if(invs.length){
    document.querySelectorAll(".inv-chip").forEach((c,i)=>c.classList.toggle("active",i===invIdx));
    const inv=invs[invIdx]||{};
    asst.textContent=`Invocation #${inv.invocation_index??invIdx+1}\n`+
      `tokens_out=${inv.completion_tokens}  seconds=${Number(inv.seconds||0).toFixed(2)}\n\n`+
      (inv.assistant_preview||"(no assistant text)");
  }
  const layerRows=currentLayerRows();
  activeLayers=layerRows;
  el("layerStrip").innerHTML=layerRows.map((row,idx)=>{
    const inBand=workspaceBand&&row.layer>=workspaceBand[0]&&row.layer<=workspaceBand[1];
    return `<button class="layer-chip${inBand?" workspace":""}" data-idx="${idx}">L${row.layer}</button>`;
  }).join("")||"<span class='empty'>No layers for this invocation</span>";
  el("layerStrip").querySelectorAll(".layer-chip").forEach(chip=>{
    chip.addEventListener("click",()=>{layerIdx=+chip.dataset.idx;renderLayer()});
  });
  if(layerRows.length) renderLayer();
  else{
    el("layerTitle").textContent="No J-space layers";
    el("repGrid").innerHTML="";
    el("trajectory").innerHTML="";
    el("sparseBox").innerHTML="";
  }
  renderChaos();
}

function renderChaos(){
  const sec=el("chaosSection");
  const box=el("chaosRadar");
  const invs=active.invocations||[];
  const inv=invs[invIdx]||{};
  const ch=inv.chaos;
  if(!ch){
    sec.style.display="none";
    box.style.display="none";
    return;
  }
  sec.style.display="block";
  box.style.display="block";
  const comp=ch.composites||{};
  const timing=ch.timing||{};
  const audit=inv.chaos_audit||{};
  const checkers=audit.checkers||{};
  const degraded=!!ch.degraded_mode;
  const flags=(ch.flags_timeline||[]).slice(0,40);
  const probes=ch.probe_layers||[10,16,22];
  const tokens=ch.tokens||[];
  // heatmap: for each probe layer, sample entropy from probes payload every ~stride
  let heat="";
  probes.forEach(L=>{
    const cells=[];
    const stride=Math.max(1, Math.floor(tokens.length/48)||1);
    for(let i=0;i<tokens.length;i+=stride){
      const row=tokens[i]||{};
      const pdata=(row.probes||{})[String(L)]||{};
      const a=pdata.A||row.A_vocab||{};
      const h=Number(a.entropy||0);
      const alpha=Math.min(1, h/4);
      const bg=`rgba(201,162,39,${0.15+0.85*alpha})`;
      cells.push(`<div class="cell" style="background:${bg}" title="step ${i} L${L} H=${h.toFixed(2)}"></div>`);
    }
    heat+=`<div class="row"><span class="ylab">L${L}</span>${cells.join("")}</div>`;
  });
  const agGrid=Object.entries(checkers).map(([k,c])=>{
    const sev=c.severity||"info";
    const fired=c.fired?"FIRED":"idle";
    const act=c.active?"active":"missing";
    return `<div class="ag-card ${sev}"><div class="id">${esc(k)}</div>
      <div>${esc(c.label||"")}</div>
      <div class="st">${act} · ${fired}<br>${esc(c.detail||"")}</div></div>`;
  }).join("");
  box.innerHTML=`
    <div class="section" style="margin-top:0">A–G checker matrix</div>
    <div class="ag-grid">${agGrid||"<span class='empty'>Rebuild dashboard for chaos_audit</span>"}</div>
    <div class="chaos-badges">
      <span class="chaos-badge">PLDS ${Number(comp.PLDS||0).toFixed(3)}</span>
      <span class="chaos-badge">FCI ${Number(comp.FCI||0).toFixed(3)}</span>
      <span class="chaos-badge">SRS ${Number(comp.SRS||0).toFixed(3)}</span>
      <span class="chaos-badge">p95 ${Number(timing.chaos_ms_per_token_p95||0).toFixed(2)} ms/tok</span>
      <span class="chaos-badge${degraded?" bad":""}">${degraded?"DEGRADED":"full cadence"}</span>
      <span class="chaos-badge warn">cliff L${(ch.curves_E||{}).cliff_layer??"—"}</span>
    </div>
    <div class="flag-timeline">${flags.length?flags.map(f=>
      `<span class="flag-chip">s${f.step}: ${(f.flags||[]).join(",")}</span>`
    ).join(""):"<span class='empty'>No flags</span>"}</div>
    <div class="chaos-heat">${heat||"<span class='empty'>No token chaos rows</span>"}</div>
  `;
}

function renderLayer(){
  const rows=currentLayerRows();
  const row=rows[layerIdx]; if(!row) return;
  const invLabel=(active.invocations||[]).length?` · inv #${(active.invocations[invIdx]||{}).invocation_index??invIdx+1}`:"";
  el("layerTitle").textContent=`Layer L${row.layer}${invLabel} — J-space top-20`;
  el("layerStrip").querySelectorAll(".layer-chip").forEach((c,i)=>c.classList.toggle("active",i===layerIdx));

  const grid=el("repGrid");
  grid.innerHTML=[
    renderReps("Mean top-20 (all positions)",row.mean_top20,false),
    renderReps("Prompt end",row.prompt_end_top20,false),
    renderReps("Generation start",row.gen_start_top20,false),
    renderReps("Generation mid",row.gen_mid_top20,false),
  ].join("");

  const traj=row.trajectory_top1||[];
  const kp=currentKeyPositions();
  el("trajectory").innerHTML=traj.map((t,i)=>{
    let mark="";
    if(i===kp.prompt_end) mark=" [prompt-end]";
    if(i===kp.gen_start) mark=" [gen-start]";
    if(i===kp.gen_mid) mark=" [gen-mid]";
    return `<span class="pos">${i}</span>${esc(t)}${mark} · `;
  }).join("");

  let sparseHtml="";
  const sp=row.sparse_prompt_end;
  if(sp&&sp.atoms){
    sparseHtml=`<div class="sparse"><b>Sparse J-space atoms (prompt end)</b> rel_mse=${sp.relative_mse?.toFixed(4)}
      ${sp.atoms.map(a=>`<span class="atom">${esc(a.token)} <small>c=${a.coeff?.toFixed(3)}</small></span>`).join("")}</div>`;
  }
  const sp2=row.sparse_gen_start;
  if(sp2&&sp2.atoms){
    sparseHtml+=`<div class="sparse"><b>Sparse J-space atoms (gen start)</b> rel_mse=${sp2.relative_mse?.toFixed(4)}
      ${sp2.atoms.map(a=>`<span class="atom">${esc(a.token)} <small>c=${a.coeff?.toFixed(3)}</small></span>`).join("")}</div>`;
  }
  el("sparseBox").innerHTML=sparseHtml;
}

loadRuns();
</script>
</body>
</html>
"""


def _load_runs(out_dir: Path) -> list[dict]:
    runs: list[dict] = []
    seen: set[str] = set()
    files = sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        if path.name.endswith("_latest.json") or path.name == "catalog.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Accept classic layer runs OR agentic runs with invocations
        has_layers = bool(data.get("layers"))
        has_inv = bool(data.get("invocations"))
        if "stamp" not in data:
            continue
        if not has_layers and not has_inv:
            continue
        key = data.get("stamp", "") + ":" + path.name
        if key in seen:
            continue
        seen.add(key)
        if "experiment" not in data:
            stem = path.stem
            data["experiment"] = stem.rsplit("_", 1)[0] if "_" in stem else stem
        data["_source_file"] = path.name
        runs.append(data)
    # Prefer agentic, then jspace_workspace, then stamp
    def sort_key(r: dict):
        kind = r.get("kind") or "legacy"
        kind_rank = 0 if kind == "aider_agentic" else (1 if kind == "jspace_workspace" else 2)
        return (kind_rank, r.get("stamp", ""))

    runs.sort(key=lambda r: r.get("stamp", ""), reverse=True)
    runs.sort(key=lambda r: 0 if r.get("kind") == "aider_agentic" else (1 if r.get("kind") == "jspace_workspace" else 2))
    return runs


def build_html(runs: list[dict], inline_fallback: bool) -> str:
    html = HTML_TEMPLATE
    if inline_fallback and runs:
        payload = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
        inline = f'<script id="runs-inline" type="application/json">{payload}</script>'
        html = html.replace("</body>", inline + "\n</body>")
    return html


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dash-dir", type=Path, default=DASH_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.dash_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(args.out_dir)

    # Attach per-invocation A–G checker audit for dashboard
    try:
        from jlens_qwen.chaos.checker_audit import audit_invocation

        for run in runs:
            for inv in run.get("invocations") or []:
                if inv.get("chaos"):
                    inv["chaos_audit"] = audit_invocation(inv["chaos"])
    except ImportError:
        pass

    data_path = args.dash_dir / "data.json"
    data_path.write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")

    index = args.dash_dir / "index.html"
    index.write_text(build_html(runs, inline_fallback=True), encoding="utf-8")

    catalog = [
        {
            "experiment": r.get("experiment"),
            "kind": r.get("kind", "legacy"),
            "stamp": r.get("stamp"),
            "layers": len(r.get("layers") or []),
            "invocations": len(r.get("invocations") or []),
            "evidence": r.get("evidence", {}).get("passed"),
            "workspace_band": r.get("workspace", {}).get("workspace_band"),
            "chaos": any(bool(inv.get("chaos")) for inv in (r.get("invocations") or [])),
            "source": r.get("_source_file"),
        }
        for r in runs
    ]
    (args.dash_dir / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"Wrote {index} ({len(runs)} run(s))")
    for r in runs:
        ws = r.get("workspace", {}).get("workspace_band")
        ev = r.get("evidence", {})
        print(
            f"  - {r.get('experiment')}: kind={r.get('kind','legacy')} "
            f"inv={len(r.get('invocations') or [])} "
            f"layers={len(r.get('layers') or [])} workspace={ws} "
            f"evidence={ev.get('passed','?')}/{ev.get('total','?')}"
        )
    print("Serve: python experiments/serve_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
