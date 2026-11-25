// el mismo frontend sirve para anywidget y para el informe sin recursos externos
const SVG="http://www.w3.org/2000/svg";
const finite=v=>typeof v==="number"&&Number.isFinite(v);
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>1-Math.pow(1-t,3);
const clock=s=>!finite(s)?"—":[Math.floor(s/3600),Math.floor(s/60)%60,Math.floor(s)%60].map((v,i)=>i?String(v).padStart(2,"0"):String(v)).join(":");
const number=v=>finite(v)?v.toFixed(4):"—";
const node=(name,attrs={})=>{const n=document.createElementNS(SVG,name);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));return n;};

class Chart {
 constructor(host,view,reduced){
  // los nodos svg se crean una vez y el carrusel solo mueve sus contenedores
  this.host=host;this.view=view;this.reduced=reduced;this.history=[];this.visible=[];this.userDomain=null;
  this.svg=node("svg",{class:"td-svg",role:"img","aria-label":view.title});
  this.grid=node("g");this.lines=node("g");this.svg.append(this.grid,this.lines);
  const id="clip-"+Math.random().toString(36).slice(2);
  this.clip=node("clipPath",{id});this.clipRect=node("rect");this.clip.append(this.clipRect);
  const defs=node("defs");defs.append(this.clip);this.svg.prepend(defs);this.lines.setAttribute("clip-path","url(#"+id+")");
  this.ticks=Array.from({length:6},()=>{const x=node("text",{class:"td-axis","text-anchor":"middle"}),y=node("text",{class:"td-axis","text-anchor":"end"}),line=node("line",{class:"td-grid"});this.grid.append(line,x,y);return{x,y,line};});
  this.paths=view.series.map(s=>{const p=node("path",{class:"td-line",stroke:s.color}),dot=node("circle",{r:3,fill:s.color});this.lines.append(p,dot);return{p,dot};});
  this.empty=node("text",{class:"td-empty","text-anchor":"middle"});this.empty.textContent="Las curvas aparecerán después de la validación";this.svg.append(this.empty);
  this.cross=node("line",{class:"td-cross",visibility:"hidden"});this.svg.append(this.cross);
  this.tip=document.createElement("div");this.tip.className="td-tip";this.tip.hidden=true;
  this.legend=document.createElement("div");this.legend.className="td-legend";
  view.series.forEach(s=>{const item=document.createElement("span"),swatch=document.createElement("i");swatch.className="td-swatch";swatch.style.background=s.color;item.append(swatch,document.createTextNode(s.label));this.legend.append(item);});
  host.append(this.svg,this.tip,this.legend);this.listeners=[];
  const listen=(event,fn,options)=>{this.svg.addEventListener(event,fn,options);this.listeners.push([event,fn,options]);};
  listen("wheel",e=>{if(!e.ctrlKey&&!e.metaKey)return;e.preventDefault();const d=this.currentDomain||this.domain(),r=this.svg.getBoundingClientRect(),x=Math.max(0,Math.min(1,(e.clientX-r.left-48)/(r.width-70))),factor=Math.exp(Math.sign(e.deltaY)*.18),center=lerp(d[0],d[1],x),span=Math.max(.5,(d[1]-d[0])*factor);this.userDomain=[center-span*x,center+span*(1-x),d[2],d[3]];cancelAnimationFrame(this.raf);this.currentDomain=this.userDomain.slice();this.draw();},{passive:false});
  listen("dblclick",()=>this.reset());
  listen("pointerdown",e=>{if(e.button!==0)return;this.drag={x:e.clientX,domain:(this.currentDomain||this.domain()).slice()};this.svg.setPointerCapture(e.pointerId);});
  listen("pointerup",()=>{this.drag=null;});listen("pointercancel",()=>{this.drag=null;});
  listen("pointermove",e=>{if(this.drag){const shift=(e.clientX-this.drag.x)/(this.width-70)*(this.drag.domain[1]-this.drag.domain[0]);this.userDomain=[this.drag.domain[0]-shift,this.drag.domain[1]-shift,...this.drag.domain.slice(2)];this.currentDomain=this.userDomain.slice();cancelAnimationFrame(this.raf);this.draw();}else this.hover(e);});
  listen("pointerleave",()=>{this.tip.hidden=true;this.cross.setAttribute("visibility","hidden");});
  this.resize=new ResizeObserver(()=>this.draw());this.resize.observe(host);
 }
 domain(){
  const values=this.view.series.flatMap(s=>this.history.map(r=>r[s.key]).filter(finite)),max=Math.max(0,...values),min=Math.min(0,...values);
  return[1,Math.max(2,...this.history.map(r=>r.epoch||1)),min,max>min?max*1.12:1];
 }
 update(history,animate=true){
  // history conserva los valores reales y visible contiene únicamente la interpolación visual
  this.history=history;
  const target=this.view.series.map(s=>history.map(r=>finite(r[s.key])?{id:r.epoch,x:r.epoch,y:r[s.key]}:null));
  const from=this.visible,startDomain=this.currentDomain||this.domain(),endDomain=this.userDomain||this.domain();
  cancelAnimationFrame(this.raf);
  // una actualización nueva parte del dibujo actual y cancela cualquier transición anterior
  const duration=animate&&!this.reduced&&from.length?500:0,start=performance.now();
  const frame=now=>{
   const t=duration?Math.min(1,(now-start)/duration):1,k=ease(t);
   this.currentDomain=endDomain.map((v,i)=>lerp(startDomain[i],v,k));
   this.visible=target.map((points,i)=>{const old=new Map((from[i]||[]).filter(Boolean).map(p=>[p.id,p]));let previous=null;return points.map(p=>{if(!p)return null;const a=old.get(p.id)||previous||p;previous=old.get(p.id)||p;return{id:p.id,x:lerp(a.x,p.x,k),y:lerp(a.y,p.y,k)};});});
   this.draw();if(t<1)this.raf=requestAnimationFrame(frame);
  };
  this.raf=requestAnimationFrame(frame);
 }
 reset(){this.userDomain=null;this.update(this.history,false);}
 draw(){
  this.width=this.host.getBoundingClientRect().width||800;
  const w=this.width,h=w<590?260:292,l=48,r=22,t=24,b=35,d=this.currentDomain||this.domain();
  this.svg.setAttribute("viewBox",`0 0 ${w} ${h}`);
  const X=x=>l+(x-d[0])/(d[1]-d[0]||1)*(w-l-r),Y=y=>h-b-(y-d[2])/(d[3]-d[2]||1)*(h-t-b);
  this.clipRect.setAttribute("x",l);this.clipRect.setAttribute("y",t);this.clipRect.setAttribute("width",Math.max(1,w-l-r));this.clipRect.setAttribute("height",h-t-b);
  this.ticks.forEach(({x,y,line},i)=>{const xv=lerp(d[0],d[1],i/5),yv=lerp(d[2],d[3],i/5);x.setAttribute("x",X(xv));x.setAttribute("y",h-12);x.textContent=Number(xv.toFixed(1));y.setAttribute("x",l-9);y.setAttribute("y",Y(yv)+3);y.textContent=yv.toFixed(d[3]<=1?2:1);line.setAttribute("x1",l);line.setAttribute("x2",w-r);line.setAttribute("y1",Y(yv));line.setAttribute("y2",Y(yv));});
  this.paths.forEach(({p,dot},i)=>{let path="",active=false,last=null;(this.visible[i]||[]).forEach(point=>{if(!point){active=false;return;}path+=(active?"L":"M")+X(point.x).toFixed(2)+","+Y(point.y).toFixed(2)+" ";active=true;last=point;});p.setAttribute("d",path);dot.setAttribute("visibility",last?"visible":"hidden");if(last){dot.setAttribute("cx",X(last.x));dot.setAttribute("cy",Y(last.y));}});
  this.empty.setAttribute("x",w/2);this.empty.setAttribute("y",h/2);this.empty.setAttribute("visibility",this.visible.some(s=>s.some(Boolean))?"hidden":"visible");
  this.cross.setAttribute("y1",t);this.cross.setAttribute("y2",h-b);
 }
 hover(e){
  if(!this.history.length)return;
  const rect=this.svg.getBoundingClientRect(),d=this.currentDomain||this.domain(),x=e.clientX-rect.left,epoch=d[0]+(x-48)/(this.width-70)*(d[1]-d[0]);
  const row=this.history.reduce((a,b)=>Math.abs(a.epoch-epoch)<Math.abs(b.epoch-epoch)?a:b);
  this.tip.textContent="Época "+row.epoch+"\n"+this.view.series.map(s=>s.label+": "+number(row[s.key])).join("\n");
  this.tip.style.left=Math.max(0,Math.min(this.width-205,x+12))+"px";this.tip.style.top="38px";this.tip.hidden=false;
  const px=48+(row.epoch-d[0])/(d[1]-d[0]||1)*(this.width-70);this.cross.setAttribute("x1",px);this.cross.setAttribute("x2",px);this.cross.setAttribute("visibility","visible");
 }
 destroy(){cancelAnimationFrame(this.raf);this.resize.disconnect();this.listeners.forEach(([event,fn,opts])=>this.svg.removeEventListener(event,fn,opts));}
}

export function mount(host,initial,options={}){
 const root=document.createElement("div");root.className="td";root.tabIndex=0;
 root.innerHTML=`<header class="td-top"><div><h2 data-model></h2><div class="td-run" data-run></div></div><div class="td-header-actions"><div class="td-status"><span class="td-dot"></span><span data-status></span></div><button class="td-theme" type="button"></button></div></header>
 <section class="td-progress"><div class="td-progress-copy"><strong data-epoch></strong><span data-batches></span></div><div class="td-bar" role="progressbar" aria-label="Progreso de la fase" aria-valuemin="0" aria-valuemax="100"><div class="td-bar-fill"></div></div></section>
 <div class="td-error" hidden></div><section class="td-cards"></section>
 <div class="td-chart-head"><div><div class="td-chart-title"></div><div class="td-chart-desc"></div></div><nav class="td-nav" aria-label="Gráficas"><button data-prev aria-label="Gráfica anterior">‹</button><span class="td-page-count"></span><button data-next aria-label="Gráfica siguiente">›</button><span class="td-pages"></span><button class="td-reset" aria-label="Restablecer zoom" title="Restablecer zoom">↺</button></nav></div>
 <div class="td-viewport"><div class="td-track"></div></div><div class="td-hint">Arrastra para desplazar · Ctrl/⌘ + rueda para zoom · Doble clic para restablecer</div>
 <div class="td-checkpoint" hidden></div><footer class="td-footer"><span data-time></span><span class="td-protocol"></span></footer><details class="td-details"><summary>Detalles del experimento y mensajes técnicos</summary><pre></pre></details>`;
 host.append(root);
 const $=s=>root.querySelector(s),reduced=matchMedia("(prefers-reduced-motion: reduce)");
 // la elección visual se guarda en el navegador y no cambia los datos del experimento
 const themeKey="parasite-training-dashboard-theme",themeButton=$(".td-theme");
 let savedTheme;
 try{savedTheme=localStorage.getItem(themeKey);}catch{}
 const notebookTheme=host.closest("[data-jp-theme-light]")?.getAttribute("data-jp-theme-light");
 const defaultTheme=notebookTheme!==undefined?(notebookTheme==="false"?"dark":"light"):(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
 function setTheme(theme){
  root.dataset.theme=theme;
  if(options.pageTheme)document.documentElement.dataset.trainingTheme=theme;
  const label=theme==="dark"?"Cambiar a modo claro":"Cambiar a modo oscuro";
  themeButton.setAttribute("aria-label",label);themeButton.setAttribute("title",label);
  themeButton.setAttribute("aria-pressed",String(theme==="dark"));
  themeButton.innerHTML=theme==="dark"?'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/></svg>':'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14A9 9 0 0 1 10 3.5 9 9 0 1 0 20.5 14Z"/></svg>';
 }
 setTheme(savedTheme==="light"||savedTheme==="dark"?savedTheme:defaultTheme);
 themeButton.onclick=()=>{setTheme(root.dataset.theme==="dark"?"light":"dark");try{localStorage.setItem(themeKey,root.dataset.theme);}catch{}};
 let state={},selected=0,tickBase=performance.now(),elapsedBase=0,charts=[],views=[],disposed=false,interval=null;
 const cards=new Map();
 [["map","mAP@50:95"],["best","Mejor mAP@50:95"],["map50","mAP@50"],["precision","Precision"],["recall","Recall"]].forEach(([key,label])=>{const c=document.createElement("div");c.className="td-card";c.innerHTML='<div class="td-label"></div><div class="td-value">—</div><div class="td-sub"></div>';c.querySelector(".td-label").textContent=label;$(".td-cards").append(c);cards.set(key,{el:c,value:c.querySelector(".td-value"),sub:c.querySelector(".td-sub"),shown:null,raf:0});});
 function animateCard(key,value){
  // interpolamos la cifra mostrada sin escribir resultados animados en el historial
  const card=cards.get(key);cancelAnimationFrame(card.raf);
  if(!finite(value)){card.value.textContent="—";card.shown=null;return;}
  const from=finite(card.shown)?card.shown:value,start=performance.now(),duration=reduced.matches||state.mode==="report"?0:400;
  const frame=now=>{if(disposed)return;const t=duration?Math.min(1,(now-start)/duration):1;card.shown=lerp(from,value,ease(t));card.value.textContent=number(card.shown);if(t<1)card.raf=requestAnimationFrame(frame);};
  card.raf=requestAnimationFrame(frame);
 }
 function navigate(index){
  selected=Math.max(0,Math.min(views.length-1,index));$(".td-track").style.transform="translateX("+(-selected*100)+"%)";
  $(".td-chart-title").textContent=views[selected]?.title||"";$(".td-chart-desc").textContent=views[selected]?.description||"";
  $(".td-page-count").textContent=(selected+1)+"/"+views.length;$("[data-prev]").disabled=selected===0;$("[data-next]").disabled=selected===views.length-1;
  charts.forEach((chart,i)=>{chart.host.classList.toggle("active",i===selected);chart.host.setAttribute("aria-hidden",i!==selected?"true":"false");});
  [...$(".td-pages").children].forEach((dot,i)=>dot.classList.toggle("active",i===selected));options.onSelect?.(selected);
 }
 $("[data-prev]").onclick=()=>navigate(selected-1);$("[data-next]").onclick=()=>navigate(selected+1);$(".td-reset").onclick=()=>charts[selected]?.reset();
 const keyboard=e=>{if(e.key==="ArrowLeft"||e.key==="ArrowRight"){e.preventDefault();e.stopPropagation();navigate(selected+(e.key==="ArrowLeft"?-1:1));}};root.addEventListener("keydown",keyboard);
 function timer(){
  const active=["preparing","training","validating"].includes(state.phase),elapsed=elapsedBase+(active?(performance.now()-tickBase)/1000:0);
  $("[data-time]").textContent="Transcurrido "+clock(elapsed)+(active&&finite(state.eta_seconds)?" · Restante ~"+clock(Math.max(0,state.eta_seconds-(elapsed-elapsedBase))):"");
 }
 function update(packet){
  const next=packet.state||packet,previousBest=state.best?.map;
  state={...state,...next};tickBase=performance.now();elapsedBase=state.elapsed_seconds||0;
  if(state.mode==="report"&&interval){clearInterval(interval);interval=null;}
  root.dataset.phase=state.phase;
  $("[data-model]").textContent=state.model_name||"Entrenamiento";$("[data-run]").textContent=state.run_name||"";
  const statuses={preparing:"Preparando",training:"Entrenando",validating:"Validando",finished:state.outcome==="early_stopped"?"Early stopped":"Completed",error:"Error",unknown:"Historial guardado"};
  $("[data-status]").textContent=statuses[state.phase]||state.phase;
  $("[data-epoch]").textContent="Época "+(state.epoch||0)+" / "+(state.total_epochs||"—");
  $("[data-batches]").textContent=state.phase==="finished"?(state.epoch||0)+" épocas registradas":(state.phase_label||"")+(state.batches?" · lote "+(state.batch||0)+" / "+state.batches:"");
  const fraction=state.phase==="finished"?1:(state.batches?Math.min(1,(state.batch||0)/state.batches):0);
  $(".td-bar-fill").style.width=(fraction*100)+"%";$('[role="progressbar"]').setAttribute("aria-valuenow",String(Math.round(fraction*100)));
  $(".td-error").hidden=!state.error;$(".td-error").textContent=state.error||"";
  const row=(state.history||[]).at(-1)||{};
  if(next.history||next.best){
   ["map","map50","precision","recall"].forEach(k=>{animateCard(k,row[k]);cards.get(k).sub.textContent=state.metric_sources?.[k]||(finite(row[k])?"Última validación":"No registrado");});
   animateCard("best",state.best?.map);cards.get("best").sub.textContent=state.best?.epoch?"Época "+state.best.epoch:"Sin validaciones";
   if(state.mode!=="report"&&finite(previousBest)&&finite(state.best?.map)&&state.best.map>previousBest){const c=cards.get("best");clearTimeout(c.flashTimer);c.el.classList.remove("record");void c.el.offsetWidth;c.el.classList.add("record");const span=document.createElement("span");span.className="td-delta";span.textContent="↑ +"+(state.best.map-previousBest).toFixed(4);c.sub.append(span);c.flashTimer=setTimeout(()=>{span.remove();c.el.classList.remove("record");},2000);}
  }
  if(next.views){
   views=next.views;
   views.slice(charts.length).forEach(view=>{const slide=document.createElement("section");slide.className="td-slide";$(".td-track").append(slide);charts.push(new Chart(slide,view,reduced.matches));const dot=document.createElement("i");dot.className="td-page-dot";$(".td-pages").append(dot);});
   navigate(selected);
  }
  if(next.history){charts.forEach(c=>c.update(state.history,packet.kind!=="restore"&&state.mode!=="report"));root.dataset.points=String(state.history.length);}
  $(".td-protocol").textContent=state.evaluation?.label||"Métricas del experimento";
  $(".td-details pre").textContent=JSON.stringify({evaluation:state.evaluation||{},config:state.config||{}},null,2)+(state.logs?.length?"\n\n"+state.logs.join("\n"):"");
  $(".td-checkpoint").hidden=!state.checkpoint;$(".td-checkpoint").textContent=state.checkpoint?"Mejor checkpoint · "+state.checkpoint:"";
  timer();
 }
 update({kind:"restore",state:initial});
 if(initial.mode!=="report")interval=setInterval(timer,1000);
 const cleanup=()=>{disposed=true;clearInterval(interval);root.removeEventListener("keydown",keyboard);charts.forEach(c=>c.destroy());cards.forEach(c=>{cancelAnimationFrame(c.raf);clearTimeout(c.flashTimer);});root.remove();};
 return{update,cleanup,get selected(){return selected;},root};
}

function render({model,el}){
 const app=mount(el,model.get("snapshot"),{onSelect:index=>{model.set("selected_view",index);model.save_changes();}});
 const pending=model.get("packet");if(pending?.state)app.update(pending);
 const changed=()=>app.update(model.get("packet"));model.on("change:packet",changed);
 return()=>{model.off("change:packet",changed);app.cleanup();};
}
export default{render};
