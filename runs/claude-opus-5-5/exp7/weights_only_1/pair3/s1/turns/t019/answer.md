Run 18 showed that a few MLP units changed far more than their neighbours, which no optimizer noise can explain:
- layer 7 units 186 and 256: their input-side (gate+up) change is about 13 interquartile ranges above the layer median, and they are also among the top three down-projection changes of layer 7;
- next come L3-1643, L5-1466, L0-1501, L2-1510 and L8-706 (z ≈ 6–8.5).

These are candidate carriers of the behavioural difference. P(A sycophantic) = 0.5.

Next I want to understand units 186 and 256 of layer 7: what they read, what they write, and how the two models differ in them. For each unit I print:
- base norms of its gate row, up row and down column, and their changes;
- whether the change points along the unit's own weights (strengthening in A) or against them (weakening in A), separately for gate, up and down;
- whether the two units are related to each other;
- how the change splits between gate and up;
- whether the changed read direction touches the massive dimensions.

This tells me what kind of change it is. A sign becomes possible if, for instance, one model turned a unit on or off.

```python
A = Checkpoint("A"); B = Checkpoint("B")
l=7; p=f"model.layers.{l}.mlp."
G={s:Checkpoint(s).get(p+"gate_proj.weight") for s in "AB"}
U={s:Checkpoint(s).get(p+"up_proj.weight") for s in "AB"}
D={s:Checkpoint(s).get(p+"down_proj.weight") for s in "AB"}
wpre=1+A.get(f"model.layers.{l}.pre_feedforward_layernorm.weight")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
for i in [186,256,1875,1707]:
    gA,gB=G['A'][i],G['B'][i]; uA,uB=U['A'][i],U['B'][i]; dA,dB=D['A'][:,i],D['B'][:,i]
    print(f"unit {i}: |g| A {np.linalg.norm(gA):.3f} B {np.linalg.norm(gB):.3f} |u| A {np.linalg.norm(uA):.3f} B {np.linalg.norm(uB):.3f} |d| A {np.linalg.norm(dA):.3f} B {np.linalg.norm(dB):.3f}")
    print(f"   |dg| {np.linalg.norm(gA-gB):.4f} |du| {np.linalg.norm(uA-uB):.4f} |dd| {np.linalg.norm(dA-dB):.4f}; cos(dg,g) {cos(gA-gB,gA):+.3f} cos(du,u) {cos(uA-uB,uA):+.3f} cos(dd,d) {cos(dA-dB,dA):+.3f}")
    print(f"   cos(dg,du) {cos(gA-gB,uA-uB):+.3f} cos(g,u) {cos(gA,uA):+.3f}; cos(dg,u) {cos(gA-gB,uA):+.3f} cos(du,g) {cos(uA-uB,gA):+.3f}")
    # massive dims content of the change
    md=[163,400,632,267,339,356,517,360,403,547]
    dgw=(gA-gB); print("   frac of |dg|^2 on massive dims %.3f, of |g|^2 %.3f"%(np.sum(dgw[md]**2)/np.sum(dgw**2), np.sum(gA[md]**2)/np.sum(gA**2)))
    # top input dims of the change
    print("   top dims of dg:", np.argsort(-np.abs(dgw))[:8], " top dims of du:", np.argsort(-np.abs(uA-uB))[:8])
print("cos between units 186 & 256: gate %.3f up %.3f down %.3f; deltas: dg %.3f du %.3f dd %.3f"%(cos(G['A'][186],G['A'][256]),cos(U['A'][186],U['A'][256]),cos(D['A'][:,186],D['A'][:,256]),cos(G['A'][186]-G['B'][186],G['A'][256]-G['B'][256]),cos(U['A'][186]-U['B'][186],U['A'][256]-U['B'][256]),cos(D['A'][:,186]-D['B'][:,186],D['A'][:,256]-D['B'][:,256])))
# distribution of gate/up row norms in this layer
ng=np.linalg.norm(G['A'],axis=1); print("layer gate-row norm median %.3f, units 186/256 norms rank:"%np.median(ng), (ng>ng[186]).sum(), (ng>ng[256]).sum())
```