import fs from 'fs'; import path from 'path'; import {pathToFileURL} from 'url';
const dir=process.argv[2];
const modUrl=pathToFileURL(path.join(dir,'burn_research.js'));
const m=await import(modUrl);
const bytes=fs.readFileSync(path.join(dir,'burn_research_bg.wasm'));
const results=[];
function test(name, fn){try{const detail=fn(); results.push({name,pass:true,detail});}catch(e){results.push({name,pass:false,error:String(e?.stack||e)});}}
function assert(c,msg){if(!c) throw new Error(msg)}
function arr(a){return Array.from(a)}

test('wasm.validate',()=>{assert(WebAssembly.validate(bytes),'WebAssembly.validate false'); return {bytes:bytes.length};});
m.initSync({module:bytes});

test('strict rejects invalid config',()=>{
 let ok=0; for(const f of [()=>m.EsOptimizer.strict(0,0,1,16,0.2,0.05),()=>m.EsOptimizer.strict(2,99,1,16,0.2,0.05),()=>m.EsOptimizer.strict(2,0,1,16,NaN,0.05)]){try{f()}catch{ok++}}
 assert(ok===3,`rejections=${ok}`); return {rejections:ok};
});

test('ES ask/tell lifecycle + cardinality',()=>{
 const o=m.EsOptimizer.strict(2,0,123,8,0.2,0.05); let pre=false,badCard=false,badFinite=false;
 try{o.tell(new Float32Array(8))}catch{pre=true}
 const c=o.ask(); assert(c.length===16,'ask length');
 try{o.tell(new Float32Array(7))}catch{badCard=true}
 const c2=o.ask(); const fit=new Float32Array(8); fit.fill(-1); fit[0]=NaN;
 try{o.tell(fit)}catch{badFinite=true}
 const c3=o.ask(); const good=new Float32Array(8); for(let i=0;i<8;i++){const x=c3[i*2],y=c3[i*2+1];good[i]=-(x*x+y*y)}
 const rep=JSON.parse(o.tell(good)); assert(o.generation()===1,'generation'); o.free();
 assert(pre&&badCard&&badFinite,'lifecycle boundary'); return {pre,badCard,badFinite,generation:1,report:rep};
});

test('mathVerifyVectors',()=>{
 const a=new Float32Array([1,2,3]);
 const same=JSON.parse(m.mathVerifyVectors(a,new Float32Array([1,2,3]),1e-6,1e-6));
 const diff=JSON.parse(m.mathVerifyVectors(a,new Float32Array([1,2,3.1]),1e-6,1e-6));
 assert(same.passed===true,'same not passed'); assert(diff.passed===false,'diff passed'); return {same,diff};
});

test('compiled ReLU + verifyFlat',()=>{
 const reg=new m.LayerRegistry(); const spec=m.AgentLayerSpec.relu(1); reg.initAgentLayer(spec);
 const b=new m.AgentGraphBuilder(2); b.addUnary(spec,0,1); b.setOutput(1); const g=b.compile(reg);
 const input=new m.WasmTensor(new Float32Array([-2,3]),new Uint32Array([1,2,1,1])); const out=g.run(reg,input); const got=arr(out.to_array());
 assert(got.length===2 && got[0]===0 && got[1]===3,`relu=${got}`);
 const proof=JSON.parse(g.verifyFlat(reg,input,new Float32Array([0,3]),1e-6,1e-6)); assert(proof.passed===true,'verifyFlat');
 input.free();out.free();g.free();b.free();spec.free();reg.free(); return {got,proof};
});

test('workspace slot lifecycle',()=>{
 const ws=new m.AgentWorkspace(3); let inputRelease=false,doubleRelease=false;
 try{ws.releaseSlot(0)}catch{inputRelease=true}
 const s=ws.reserveSlot('audit'); assert(s===1,'slot1 expected'); ws.releaseSlot(s);
 try{ws.releaseSlot(s)}catch{doubleRelease=true}
 const s2=ws.reserveSlot('audit2'); assert(s2===1,'reuse slot1');
 assert(inputRelease&&doubleRelease,'slot rejection'); ws.free(); return {inputRelease,doubleRelease,reused:s2};
});

test('workspace init atomic rollback',()=>{
 const ws=new m.AgentWorkspace(2), reg=new m.LayerRegistry(), b=new m.AgentGraphBuilder(1);
 const id=ws.reserveLayerId(reg,'relu'); const spec=m.AgentLayerSpec.relu(id); const before=ws.snapshot(); const steps=b.numSteps(); const params=reg.totalParams(); let err='';
 try{m.workspaceInitUnary(ws,b,reg,spec,0,'relu')}catch(e){err=String(e)}
 assert(err,'expected failure'); assert(ws.snapshot()===before,'workspace mutated'); assert(b.numSteps()===steps,'builder mutated'); assert(reg.totalParams()===params,'registry params mutated'); assert(!reg.layerExists(spec.layerType(),id),'layer leaked');
 spec.free(); b.free(); reg.free(); ws.free(); return {err,rollback:true};
});

test('layout preflight incompatible is atomic',()=>{
 const ws=new m.AgentWorkspace(8),reg=new m.LayerRegistry(),b=new m.AgentGraphBuilder(8);
 const eid=ws.reserveLayerId(reg,'emb'); const emb=m.AgentLayerSpec.embedding(eid,16,4); const eslot=m.workspaceInitUnary(ws,b,reg,emb,0,'emb');
 const lid=ws.reserveLayerId(reg,'lin'); const lin=m.AgentLayerSpec.linear(lid,4,2,true); const before=ws.snapshot(), steps=b.numSteps(), params=reg.totalParams(); let err='';
 try{m.workspaceInitUnary(ws,b,reg,lin,eslot,'lin')}catch(e){err=String(e)}
 const compat=m.agentLayoutCompatibility(emb,lin);
 assert(compat==='incompatible','compat'); assert(err.includes('incompatible'),'no incompat error'); assert(ws.snapshot()===before,'workspace mutated'); assert(b.numSteps()===steps,'steps mutated'); assert(reg.totalParams()===params,'params mutated'); assert(!reg.layerExists(lin.layerType(),lid),'consumer leaked');
 emb.free();lin.free();b.free();reg.free();ws.free(); return {compat,err,rollback:true};
});

function esReplay(){
 const o=m.EsOptimizer.strict(2,1,777,16,0.4); const trace=[];
 for(let gen=0;gen<12;gen++){const c=o.ask(); const fit=new Float32Array(16); for(let i=0;i<16;i++){const x=c[i*2],y=c[i*2+1]; fit[i]=-((x-1.25)**2+(y+0.75)**2);} trace.push({ask:arr(c),report:JSON.parse(o.tell(fit)),mean:arr(o.mean()),best:arr(o.best())});}
 o.free(); return JSON.stringify(trace);
}
test('deterministic ES replay',()=>{const a=esReplay(),b=esReplay();assert(a===b,'replay mismatch');return {exact:true,chars:a.length};});

test('objective shift exposes lifetime-global best',()=>{
 const o=m.EsOptimizer.strict(1,1,99,32,0.7);
 for(let g=0;g<30;g++){const c=o.ask(); const f=new Float32Array(32);for(let i=0;i<32;i++)f[i]=100-(c[i]+2)**2;o.tell(f)}
 const bestA=o.best()[0], meanA=o.mean()[0];
 for(let g=0;g<50;g++){const c=o.ask(); const f=new Float32Array(32);for(let i=0;i<32;i++)f[i]=-((c[i]-2)**2);o.tell(f)}
 const bestAfter=o.best()[0], meanAfter=o.mean()[0]; const report=JSON.parse(o.report()); o.free();
 assert(Math.abs(meanAfter-2)<1.0,`mean did not adapt: ${meanAfter}`); assert(Math.abs(bestAfter+2)<1.0,`global best unexpectedly reset: ${bestAfter}`);
 return {bestA,meanA,bestAfter,meanAfter,report_note:'best remains lifetime-global across incomparable objectives'};
});

test('program capabilities declare structural identity + required binding',()=>{
 const caps=JSON.parse(m.programCapabilities());
 assert(caps.identity_schema==='burn-research.program-identity.v1','identity schema');
 assert(caps.plan==='programPlan','plan discovery');
 assert(caps.identity==='programIdentity','identity discovery');
 assert(caps.binding_validation==='validateRegistryBinding','binding discovery');
 assert(caps.execution_binding==='required','execution binding must be required');
 assert(caps.mutable_state_in_identity===false,'mutable state must stay outside structural identity');
 return caps;
});

test('programPlan replay preserves exact structural identity',()=>{
 const reg=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(7,2,2,true); reg.initAgentLayer(spec);
 const b=new m.AgentGraphBuilder(2); b.addUnary(spec,0,1); b.setOutput(1); const g=b.compile(reg);
 const plan=g.programPlan(); const identity=g.programIdentity(); const replay=reg.compileGraph(plan);
 assert(JSON.stringify(arr(replay.programPlan()))===JSON.stringify(arr(plan)),'plan replay mismatch');
 assert(replay.programIdentity()===identity,'identity replay mismatch');
 replay.validateRegistryBinding(reg);
 const detail={plan:arr(plan),identity};
 replay.free(); g.free(); b.free(); spec.free(); reg.free(); return detail;
});

test('compatible weight update preserves identity and remains executable',()=>{
 const reg=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(7,2,2,true); reg.initAgentLayer(spec);
 const b=new m.AgentGraphBuilder(2); b.addUnary(spec,0,1); b.setOutput(1); const g=b.compile(reg);
 const type=spec.layerType(), identity=g.programIdentity();
 const weights=reg.getWeightsFlat(7,type); const zeros=new Float32Array(weights.length); reg.setWeightsFlat(7,type,zeros);
 const input=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1]));
 const outZero=g.run(reg,input); const a=arr(outZero.to_array());
 const ones=new Float32Array(weights.length); ones.fill(1); reg.setWeightsFlat(7,type,ones);
 g.validateRegistryBinding(reg); const outOne=g.run(reg,input); const z=arr(outOne.to_array());
 assert(g.programIdentity()===identity,'structural identity changed after compatible weight update');
 assert(a.length===z.length && a.some((v,i)=>v!==z[i]),`updated weights not observed: ${a} -> ${z}`);
 const detail={identity,zero_output:a,updated_output:z};
 outZero.free();outOne.free();input.free();g.free();b.free();spec.free();reg.free();return detail;
});

test('structural Registry drift rejects old graph and recompile gets new identity',()=>{
 const reg=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(7,2,2,true); reg.initAgentLayer(spec);
 const b=new m.AgentGraphBuilder(2); b.addUnary(spec,0,1); b.setOutput(1); const g=b.compile(reg);
 const plan=g.programPlan(), oldIdentity=g.programIdentity(); const replacement=m.AgentLayerSpec.linear(7,2,3,true); reg.initAgentLayer(replacement);
 let bindErr='', runErr=''; try{g.validateRegistryBinding(reg)}catch(e){bindErr=String(e)}
 const input=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1])); try{g.run(reg,input)}catch(e){runErr=String(e)}
 assert(bindErr.includes('structural identity mismatch'),`binding did not reject drift: ${bindErr}`);
 assert(runErr.includes('structural identity mismatch'),`run did not reject drift: ${runErr}`);
 assert(g.programIdentity()===oldIdentity,'compiled identity mutated after Registry drift');
 const replay=reg.compileGraph(plan), newIdentity=replay.programIdentity(); const out=replay.run(reg,input); const got=arr(out.to_array());
 assert(newIdentity!==oldIdentity,'recompile did not produce new structural identity');
 assert(got.length===3,`recompiled graph did not observe new structure: ${got}`);
 const detail={bindErr,runErr,oldIdentity,newIdentity,recompiled_output_len:got.length};
 out.free();replay.free();input.free();replacement.free();g.free();b.free();spec.free();reg.free();return detail;
});

test('missing referenced layer rejects run and verifyFlat before execution',()=>{
 const reg=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(7,2,2,true); reg.initAgentLayer(spec);
 const b=new m.AgentGraphBuilder(2); b.addUnary(spec,0,1); b.setOutput(1); const g=b.compile(reg);
 const input=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1]));
 assert(reg.destroyLayer(7,spec.layerType())===true,'destroy failed'); let runErr='',verifyErr='';
 try{g.run(reg,input)}catch(e){runErr=String(e)}
 try{g.verifyFlat(reg,input,new Float32Array([0,0]),1e-6,1e-6)}catch(e){verifyErr=String(e)}
 assert(runErr.includes('not found'),`run missing-layer rejection absent: ${runErr}`);
 assert(verifyErr.includes('not found'),`verifyFlat bypassed binding guard: ${verifyErr}`);
 const detail={runErr,verifyErr}; input.free();g.free();b.free();spec.free();reg.free();return detail;
});

let asyncInit;
try{ const m2=await import(modUrl.href+'?asyncprobe=1'); await m2.default(); asyncInit={ok:true}; }catch(e){ asyncInit={ok:false,error:String(e)}; }
results.push({name:'default async init local file',pass:asyncInit.ok,expected_portability_caveat:!asyncInit.ok,detail:asyncInit});

const requiredFailures=results.filter(x=>!x.pass&&!x.expected_portability_caveat);
const knownCaveats=results.filter(x=>!x.pass&&x.expected_portability_caveat);
const requiredTotal=results.length-knownCaveats.length;
const requiredPassCount=results.filter(x=>x.pass).length;
const verdict=requiredFailures.length===0?(knownCaveats.length?'PASS_WITH_KNOWN_LIMITATIONS':'PASS'):'FAIL';
console.log(JSON.stringify({verdict,requiredPassCount,requiredTotal,total:results.length,knownCaveats:knownCaveats.map(x=>x.name),results},null,2));
if(requiredFailures.length) process.exitCode=1;
