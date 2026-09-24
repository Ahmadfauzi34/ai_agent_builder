import fs from 'fs'; import path from 'path'; import {pathToFileURL} from 'url';
import {createHash} from 'node:crypto';
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

test('multi-input graph plan preflights all external slots before execution',()=>{
 const reg=new m.LayerRegistry(), add=m.AgentLayerSpec.add(21); reg.initAgentLayer(add);
 const b=new m.AgentGraphBuilder(3); b.addBinary(add,0,1,2); b.setOutput(2);
 const plan=b.multiInputPlanV1();
 plan.addInputPort(0,'observation',1,2,1,1,'feature_axis1_singleton',true,2n);
 plan.addInputPort(1,'state',1,2,1,1,'feature_axis1_singleton',true,3n);
 assert(JSON.parse(plan.toJSON()).schema_id==='burn-research.multi-input-graph-plan.v1','multi-input plan schema');
 const encoded=plan.toBytes(), replay=m.MultiInputGraphPlan.fromBytes(encoded);
 assert(JSON.stringify(arr(replay.toBytes()))===JSON.stringify(arr(encoded)),'multi-input plan round-trip');
 const graph=reg.compileMultiInputGraph(replay), incomplete=new m.MultiInputInputBundle(replay);
 const missing=JSON.parse(graph.preflight(reg,incomplete)); assert(missing.ready===false,'missing input passed preflight');
 let missingErr=''; try{graph.run(reg,incomplete)}catch(e){missingErr=String(e)}
 assert(missingErr.includes('preflight failed'),'execution started with an unbound port');
 const left=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1]));
 const right=new m.WasmTensor(new Float32Array([3,4]),new Uint32Array([1,2,1,1]));
 incomplete.bindInput(0,left,'state','feature_axis1_singleton','sensor-a',2n,'sha256:obs');
 incomplete.bindInput(1,right,'state','feature_axis1_singleton','memory-b',3n,'sha256:state');
 const mismatch=JSON.parse(graph.preflight(reg,incomplete)); assert(mismatch.ready===false,'role mismatch passed preflight');
 let mismatchErr=''; try{graph.run(reg,incomplete)}catch(e){mismatchErr=String(e)}
 assert(mismatchErr.includes('preflight failed'),'mismatched input was executed');
 incomplete.clearInput(0);
 incomplete.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',2n,'sha256:obs');
 const ready=JSON.parse(graph.preflight(reg,incomplete)); assert(ready.ready===true,'valid bundle failed preflight');
 const out=graph.run(reg,incomplete), got=arr(out.to_array()); assert(JSON.stringify(got)==='[4,6]',`output=${got}`);
 const verification=JSON.parse(graph.verifyFlat(reg,incomplete,new Float32Array([4,6]),1e-6,1e-6));
 assert(verification.verification.passed===true,'multi-input verifyFlat failed');
 incomplete.clearInput(0);
 const wrongShape=new m.WasmTensor(new Float32Array([1,2,3]),new Uint32Array([1,3,1,1]));
 incomplete.bindInput(0,wrongShape,'observation','feature_axis1_singleton','sensor-a',2n,'sha256:obs');
 const shapeMismatch=JSON.parse(graph.preflight(reg,incomplete));
 assert(shapeMismatch.ready===false && shapeMismatch.inputs.ports[0].shape_matches===false,'shape mismatch passed preflight');
 let shapeErr=''; try{graph.run(reg,incomplete)}catch(e){shapeErr=String(e)}
 assert(shapeErr.includes('preflight failed'),'wrong-shaped input was executed');
 incomplete.clearInput(0); incomplete.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',2n,'sha256:obs');
 const otherAdd=m.AgentLayerSpec.add(22); reg.initAgentLayer(otherAdd);
 const otherBuilder=new m.AgentGraphBuilder(3); otherBuilder.addBinary(otherAdd,0,1,2); otherBuilder.setOutput(2);
 const otherPlan=otherBuilder.multiInputPlanV1();
 otherPlan.addInputPort(0,'observation',1,2,1,1,'feature_axis1_singleton',true,2n);
 otherPlan.addInputPort(1,'state',1,2,1,1,'feature_axis1_singleton',true,3n);
 const otherBundle=new m.MultiInputInputBundle(otherPlan);
 otherBundle.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',2n,'sha256:obs');
 otherBundle.bindInput(1,right,'state','feature_axis1_singleton','memory-b',3n,'sha256:state');
 const planMismatch=JSON.parse(graph.preflight(reg,otherBundle));
 assert(planMismatch.bundle_plan_matches===false,'bundle from another exact plan was accepted');
 const replacement=m.AgentLayerSpec.sub(21); reg.initAgentLayer(replacement);
 const drift=JSON.parse(graph.preflight(reg,incomplete));
 assert(drift.registry_binding_current===false && drift.ready===false,'registry drift passed preflight');
 const caps=JSON.parse(m.multiInputGraphCapabilities()); assert(caps.execution_authorized_by_preflight===false,'preflight must not claim authorization');
 const detail={missing:missing.ready,mismatch:mismatch.ready,shape_mismatch:shapeMismatch.ready,ready:ready.ready,plan_mismatch:planMismatch.bundle_plan_matches,registry_current:drift.registry_binding_current,required_slots:arr(graph.requiredInputSlots()),got,verification:verification.verification};
 replacement.free();otherBundle.free();otherPlan.free();otherBuilder.free();otherAdd.free();wrongShape.free();out.free();left.free();right.free();incomplete.free();graph.free();replay.free();plan.free();b.free();add.free();reg.free(); return detail;
});

test('baseline-candidate Burn verification receipt binds exact cases and mutable state',()=>{
 function machine(spec){
  const registry=new m.LayerRegistry(); registry.initAgentLayer(spec);
  const builder=new m.AgentGraphBuilder(3); builder.addBinary(spec,0,1,2); builder.setOutput(2);
  const plan=builder.multiInputPlanV1();
  plan.addInputPort(0,'observation',1,2,1,1,'feature_axis1_singleton',false,0n);
  plan.addInputPort(1,'state',1,2,1,1,'feature_axis1_singleton',false,0n);
  return {registry,builder,plan,graph:registry.compileMultiInputGraph(plan),spec};
 }
 function inputs(plan,leftValues=[1,2],rightValues=[3,4]){
  const bundle=new m.MultiInputInputBundle(plan);
  const left=new m.WasmTensor(new Float32Array(leftValues),new Uint32Array([1,2,1,1]));
  const right=new m.WasmTensor(new Float32Array(rightValues),new Uint32Array([1,2,1,1]));
  bundle.bindInput(0,left,'observation','feature_axis1_singleton','test',0n,'');
  bundle.bindInput(1,right,'state','feature_axis1_singleton','test',0n,'');
  left.free();right.free();return bundle;
 }
 const base=machine(m.AgentLayerSpec.add(42)), same=machine(m.AgentLayerSpec.add(43));
 const different=machine(m.AgentLayerSpec.sub(42));
 const equalCases=new m.MultiInputVerificationCases(base.graph,same.graph);
 const left=inputs(base.plan),right=inputs(same.plan);
 assert(equalCases.addCase(left,right)===1,'case cardinality');
 const equal=JSON.parse(equalCases.verify(base.registry,base.graph,same.registry,same.graph,0,0));
 assert(equal.equivalent===true&&equal.tested_vector_count===1&&equal.compared_f32_count===2,'equal program receipt');
 assert(equal.promotion_authorized===false&&equal.baseline_program_identity.input_plan_hex!==equal.candidate_program_identity.input_plan_hex,'identity or authority');
 const stateBytes=m.exportMultiInputProgramBundle(base.graph,base.registry,true);
 assert(equal.baseline_state_checkpoint_bytes_sha256===`sha256:${createHash('sha256').update(stateBytes).digest('hex')}`,'state digest mismatch');
 const receiptBody={...equal}; delete receiptBody.receipt_digest;
 assert(equal.receipt_digest===`sha256:${createHash('sha256').update(JSON.stringify(receiptBody)).digest('hex')}`,'receipt body digest mismatch');
 const unequalCases=new m.MultiInputVerificationCases(base.graph,different.graph);
 const wrong=inputs(different.plan,[1,3]);
 let rejected=false;try{unequalCases.addCase(left,wrong)}catch{rejected=true}
 assert(rejected&&unequalCases.caseCount()===0,'mismatched test vector unexpectedly accepted');
 wrong.free();
 const input=inputs(different.plan); unequalCases.addCase(left,input);
 const unequal=JSON.parse(unequalCases.verify(base.registry,base.graph,different.registry,different.graph,0,0));
 assert(unequal.equivalent===false&&unequal.first_failure_case_index===0&&unequal.max_abs_error===8,'changed operator passed');
 assert(unequal.cases[0].input_sha256===equal.cases[0].input_sha256,'identical input vectors changed digest');
 const replacement=m.AgentLayerSpec.mul(42); different.registry.initAgentLayer(replacement);
 let stale=false; try{unequalCases.verify(base.registry,base.graph,different.registry,different.graph,0,0)}catch{stale=true}
 assert(stale,'stale candidate registry issued receipt');
 replacement.free();input.free();unequalCases.free();right.free();left.free();equalCases.free();
 for(const item of [base,same,different]){item.graph.free();item.plan.free();item.builder.free();item.spec.free();item.registry.free()}
 return {equal:equal.equivalent,unequal:unequal.equivalent,stale_rejected:stale,case_count:equal.tested_vector_count};
});

test('graph mutation transaction stages, verifies and commits only an issued receipt',()=>{
 const reg=new m.LayerRegistry(),add=m.AgentLayerSpec.add(61);reg.initAgentLayer(add);
 const builder=new m.AgentGraphBuilder(4);builder.addBinary(add,0,1,2);builder.setOutput(2);
 const plan=builder.multiInputPlanV1();
 for(const [slot,role] of [[0,'observation'],[1,'state']])plan.addInputPort(slot,role,1,2,1,1,'feature_axis1_singleton',false,0n);
 const graph=reg.compileMultiInputGraph(plan),original=m.exportMultiInputProgramBundle(graph,reg,true);
 const state=`sha256:${createHash('sha256').update(original).digest('hex')}`,identity=graph.programIdentity();
 const tx=new m.GraphMutationTransaction(reg,graph,identity,state);
 tx.removeStep(0);let bad=false;try{tx.stageCandidate()}catch{bad=true}
 assert(bad,'empty candidate compiled');
 const next=m.AgentLayerSpec.add(62);tx.insertStep(0,next,0,1,3);tx.setOutput(3);
 const stage=JSON.parse(tx.stageCandidate());assert(stage.promotion_authorized===false,'stage authorized promotion');
 assert(Buffer.compare(Buffer.from(original),Buffer.from(m.exportMultiInputProgramBundle(graph,reg,true)))===0,'staging changed baseline');
 const baseReg=new m.LayerRegistry(),candidateReg=new m.LayerRegistry();
 const baseGraph=m.importMultiInputProgramBundle(baseReg,tx.baselineBundle());
 const candidateGraph=m.importMultiInputProgramBundle(candidateReg,tx.candidateBundle());
 const candidatePlan=m.MultiInputGraphPlan.fromBytes(candidateGraph.inputPlanV1());
 function inputs(p){const b=new m.MultiInputInputBundle(p);
  for(const [slot,role,values] of [[0,'observation',[1,2]],[1,'state',[3,4]]]){
   const tensor=new m.WasmTensor(new Float32Array(values),new Uint32Array([1,2,1,1]));
   b.bindInput(slot,tensor,role,'feature_axis1_singleton','test',0n,'');tensor.free();
  }return b;}
 const cases=new m.MultiInputVerificationCases(baseGraph,candidateGraph);
 const inputA=inputs(plan),inputB=inputs(candidatePlan);cases.addCase(inputA,inputB);
 const receipt=JSON.parse(tx.verifyCases(cases,0,0));assert(receipt.equivalent,'same operator failed comparison');
 let forged=false,denied=false;
 try{tx.commitByReceipt(reg,graph,identity,state,'sha256:forged',true)}catch{forged=true}
 try{tx.commitByReceipt(reg,graph,identity,state,receipt.receipt_digest,false)}catch{denied=true}
 assert(forged&&denied,'receipt or authorization bypassed');
 const promoted=tx.commitByReceipt(reg,graph,identity,state,receipt.receipt_digest,true);
 assert(promoted.programIdentity()===candidateGraph.programIdentity(),'committed identity mismatch');
 assert(Buffer.compare(Buffer.from(m.exportMultiInputProgramBundle(promoted,reg,true)),Buffer.from(tx.candidateBundle()))===0,'committed state mismatch');
 for(const item of [inputA,inputB,cases,candidatePlan,candidateGraph,baseGraph,baseReg,candidateReg,promoted,tx,next,graph,plan,builder,add,reg])item.free();
 return {stage_identity_changed:JSON.parse(identity).input_plan_hex!==stage.candidate_program_identity.input_plan_hex,forged_rejected:forged,unauthorized_rejected:denied,verified:receipt.equivalent};
});

test('semantic ingress v2 maps logical ports and gates source before run',()=>{
 const reg=new m.LayerRegistry(), add=m.AgentLayerSpec.add(31); reg.initAgentLayer(add);
 const b=new m.AgentGraphBuilder(3); b.addBinary(add,0,1,2); b.setOutput(2);
 const plan=b.multiInputPlanV1();
 plan.addInputPort(0,'observation',1,2,1,1,'feature_axis1_singleton',true,2n);
 plan.addInputPort(1,'state',1,2,1,1,'feature_axis1_singleton',true,3n);
 const graph=reg.compileMultiInputGraph(plan), bundle=new m.MultiInputInputBundle(plan);
 const ingress=new m.SemanticIngressManifestV2(plan);
 assert(JSON.parse(m.semanticIngressManifestV2Capabilities()).scope.manifest_class==='SemanticIngressManifestV2','v2 capability');
 assert(JSON.parse(ingress.status(reg,graph,bundle)).ports[0].status==='runtime_backing_missing','unmapped port invisible');
 ingress.addRuntimePort('observation',0,'sensor-a'); ingress.addRuntimePort('memory',1,'memory-b');
 ingress.addDeferredPort('objective','x-objective',false);
 const spec=new m.InputPortConsumerSpec('state-reader','["state"]',false,true,3n);
 const unknown=JSON.parse(ingress.consumerCompatibility(1,bundle,spec));
 assert(unknown.status==='unknown'&&unknown.compatible===null,'unbound consumer did not report unknown');
 const left=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1]));
 const right=new m.WasmTensor(new Float32Array([3,4]),new Uint32Array([1,2,1,1]));
 bundle.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',2n,'obs');
 bundle.bindInput(1,right,'state','feature_axis1_singleton','wrong-source',3n,'state');
 assert(JSON.parse(graph.preflight(reg,bundle)).ready===true,'graph preflight failed for valid tensor contracts');
 const wrongSource=JSON.parse(ingress.status(reg,graph,bundle));
 assert(wrongSource.ready===false&&wrongSource.ports[1].status==='source_mismatch','bridge accepted wrong source');
 let gated=false; try{ingress.run(reg,graph,bundle)}catch{gated=true}
 assert(gated,'source mismatch reached execution');
 bundle.clearInput(1);
 bundle.bindInput(1,right,'state','feature_axis1_singleton','memory-b',3n,'state');
 const ready=JSON.parse(ingress.status(reg,graph,bundle));
 assert(ready.ready===true&&ready.execution_authorized===false,'valid v2 bridge readiness incorrect');
 assert(JSON.parse(ingress.inputPortStatus(1,bundle)).port.logical_port_id==='memory','slot discovery');
 assert(JSON.parse(ingress.consumerCompatibility(1,bundle,spec)).compatible===true,'state consumer compatibility');
 const out=ingress.run(reg,graph,bundle),got=arr(out.to_array());
 assert(JSON.stringify(got)==='[4,6]',`ingress output=${got}`);
 const proof=JSON.parse(ingress.verifyFlat(reg,graph,bundle,new Float32Array([4,6]),1e-6,1e-6));
 assert(proof.ingress.ready===true&&proof.reference.verification.passed===true,'bridge verification failed');
 const required=new m.SemanticIngressManifestV2(plan);
 required.addRuntimePort('observation',0,'sensor-a');required.addRuntimePort('memory',1,'memory-b');
 required.addDeferredPort('objective','x-objective',true);
 assert(JSON.parse(required.status(reg,graph,bundle)).runtime_coverage_complete===false,'required deferred port was treated as executable');
 const detail={wrong_source:wrongSource.ports[1].status,ready:ready.ready,consumer:'compatible',got,proof:proof.reference.verification};
 required.free();out.free();right.free();left.free();spec.free();ingress.free();bundle.free();graph.free();plan.free();b.free();add.free();reg.free();return detail;
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

test('program bundle capabilities declare atomic replace + separate state',()=>{
 const caps=JSON.parse(m.programBundleCapabilities());
 assert(caps.schema==='burn-research.program-bundle.v1','bundle schema');
 assert(caps.structural_identity==='burn-research.program-identity.v1','bundle identity schema');
 assert(caps.target_registry==='atomic_replace_on_success','target replacement semantics');
 assert(caps.import_commit==='atomic_after_identity_validation','atomic commit semantics');
 assert(caps.mutable_state==='optional_separate_section','state separation');
 return caps;
});

test('program bundle state round-trip preserves identity and output',()=>{
 const source=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(77,2,2,true); source.initAgentLayer(spec);
 const builder=new m.AgentGraphBuilder(2); builder.addUnary(spec,0,1); builder.setOutput(1); const graph=builder.compile(source);
 const type=spec.layerType(); const weights=source.getWeightsFlat(77,type); const fixed=new Float32Array(weights.length); for(let i=0;i<fixed.length;i++)fixed[i]=(i+1)*0.125; source.setWeightsFlat(77,type,fixed);
 const input=new m.WasmTensor(new Float32Array([1.5,-0.5]),new Uint32Array([1,2,1,1])); const before=graph.run(source,input); const expected=arr(before.to_array());
 const bundle=m.exportProgramBundle(graph,source,true); const oldIdentity=graph.programIdentity();
 const target=new m.LayerRegistry(); const old=m.AgentLayerSpec.relu(99); target.initAgentLayer(old);
 const imported=m.importProgramBundle(target,bundle); const after=imported.run(target,input); const got=arr(after.to_array());
 assert(imported.programIdentity()===oldIdentity,'bundle identity changed');
 assert(JSON.stringify(arr(imported.programPlan()))===JSON.stringify(arr(graph.programPlan())),'bundle plan changed');
 assert(JSON.stringify(got)===JSON.stringify(expected),`bundle output mismatch: ${expected} -> ${got}`);
 assert(!target.layerExists(old.layerType(),old.layerId()),'successful import did not replace prior registry');
 imported.validateRegistryBinding(target);
 const detail={bytes:bundle.length,identity:oldIdentity,output:got};
 after.free();imported.free();old.free();before.free();input.free();graph.free();builder.free();spec.free();source.free();target.free();return detail;
});

test('multi-input ProgramBundle restores mutable layer state and exact plan',()=>{
 const source=new m.LayerRegistry(), add=m.AgentLayerSpec.add(81), linear=m.AgentLayerSpec.linear(82,2,2,true);
 source.initAgentLayer(add); source.initAgentLayer(linear);
 const builder=new m.AgentGraphBuilder(4); builder.addBinary(add,0,1,2); builder.addUnary(linear,2,3); builder.setOutput(3);
 const plan=builder.multiInputPlanV1();
 plan.addInputPort(0,'observation',1,2,1,1,'feature_axis1_singleton',true,1n);
 plan.addInputPort(1,'state',1,2,1,1,'feature_axis1_singleton',true,1n);
 const graph=source.compileMultiInputGraph(plan), identity=graph.programIdentity();
 const weights=source.getWeightsFlat(82,linear.layerType()); for(let i=0;i<weights.length;i++)weights[i]=(i+1)*0.125;
 source.setWeightsFlat(82,linear.layerType(),weights);
 assert(graph.programIdentity()===identity,'mutable layer state changed structural identity');
 const left=new m.WasmTensor(new Float32Array([1,2]),new Uint32Array([1,2,1,1]));
 const right=new m.WasmTensor(new Float32Array([3,4]),new Uint32Array([1,2,1,1]));
 const inputBundle=new m.MultiInputInputBundle(plan);
 inputBundle.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',1n,'obs');
 inputBundle.bindInput(1,right,'state','feature_axis1_singleton','memory-b',1n,'state');
 const output=graph.run(source,inputBundle), expected=arr(output.to_array());
 const bytes=m.exportMultiInputProgramBundle(graph,source,true);
 assert(String.fromCharCode(...bytes.slice(0,8))==='BRMIBNDL','multi-input bundle magic');
 const target=new m.LayerRegistry(), previous=m.AgentLayerSpec.relu(99); target.initAgentLayer(previous);
 const imported=m.importMultiInputProgramBundle(target,bytes);
 const restoredPlan=m.MultiInputGraphPlan.fromBytes(imported.inputPlanV1());
 const restoredBundle=new m.MultiInputInputBundle(restoredPlan);
 restoredBundle.bindInput(0,left,'observation','feature_axis1_singleton','sensor-a',1n,'obs');
 restoredBundle.bindInput(1,right,'state','feature_axis1_singleton','memory-b',1n,'state');
 const restoredOutput=imported.run(target,restoredBundle), got=arr(restoredOutput.to_array());
 assert(imported.programIdentity()===identity,'multi-input bundle identity changed');
 assert(JSON.stringify(arr(imported.inputPlanV1()))===JSON.stringify(arr(graph.inputPlanV1())),'multi-input contract bytes changed');
 assert(JSON.stringify(got)===JSON.stringify(expected),`multi-input state output mismatch: ${expected} -> ${got}`);
 assert(!target.layerExists(previous.layerType(),previous.layerId()),'successful multi-input import did not replace target registry');
 const changed=new Float32Array(weights); changed[0]+=1; source.setWeightsFlat(82,linear.layerType(),changed);
 const changedBytes=m.exportMultiInputProgramBundle(graph,source,true);
 assert(graph.programIdentity()===identity,'changed weights changed structural identity');
 assert(JSON.stringify(arr(changedBytes))!==JSON.stringify(arr(bytes)),'changed mutable state did not change checkpoint bytes');
 const caps=JSON.parse(m.multiInputProgramBundleCapabilities());
 assert(caps.structural_identity==='burn-research.multi-input-program-identity.v1'
   && caps.state_integrity==='no_signature_or_authentication' && caps.authorization===false,'multi-input checkpoint authority contract');
 const detail={bundle_bytes:bytes.length,identity,output:got,mutable_state_changed_checkpoint:true,identity_stable:true};
 restoredOutput.free();restoredBundle.free();restoredPlan.free();imported.free();previous.free();target.free();
 output.free();inputBundle.free();left.free();right.free();graph.free();plan.free();builder.free();add.free();linear.free();source.free();return detail;
});

test('corrupt program bundle fails atomically',()=>{
 const source=new m.LayerRegistry(); const spec=m.AgentLayerSpec.linear(77,2,2,true); source.initAgentLayer(spec);
 const builder=new m.AgentGraphBuilder(2); builder.addUnary(spec,0,1); builder.setOutput(1); const graph=builder.compile(source); const bundle=m.exportProgramBundle(graph,source,true); const corrupt=bundle.slice(0,bundle.length-1);
 const target=new m.LayerRegistry(); const existing=m.AgentLayerSpec.relu(99); target.initAgentLayer(existing); let err='';
 try{m.importProgramBundle(target,corrupt)}catch(e){err=String(e)}
 assert(err,'corrupt bundle unexpectedly imported'); assert(target.layerExists(existing.layerType(),existing.layerId()),'target registry mutated on failed import');
 const detail={err,target_preserved:true}; existing.free();graph.free();builder.free();spec.free();source.free();target.free();return detail;
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
