Run 24 showed that the sign anchors are unreliable in the late layers. The two GeGLU constraints disagree (for example s1 = −6 against s2 = +2 in L17), so the late-layer κ results cannot be trusted. Only the layer-7 anchor of the two strongly active units 186 and 256 is solid. Note also that q-norm removes changes in query magnitude, so only direction changes matter. P(A sycophantic) = 0.5.

The whole problem is symmetric between "copy the guess" and "recall the answer":
- the sycophantic model is pushed to copy more and to recall less;
- the correct model is pushed the other way.

The layer-7 result, that A turns down units 186 and 256 at the answer position, decides the question once I know what these units feed. If their output drives copy routing (it is read by the queries of copy heads), turning them down suppresses copying, which points to A being correct. If it feeds the recall/answer-writing path (late MLPs, the logits), turning it down suppresses recall, which points to A being sycophantic.

So I measure how strongly each later component reads the base down-vectors of units 186 and 256 (through the post-FF scaling of layer 7 and the input scaling of the reader). I compare against random directions with the same scaling, for:
- each head's query, key and value;
- each later MLP's gate/up;
- the final norm (direct logit path).

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(1)
hd=256
p7="model.layers.7."
wpf7=1+A.get(p7+"post_feedforward_layernorm.weight")
D7=A.get(p7+"mlp.down_proj.weight")
dirs={"d186":wpf7*D7[:,186], "d256":wpf7*D7[:,256], "dd186":wpf7*(D7[:,186]-B.get(p7+"mlp.down_proj.weight")[:,186])}
rand=[wpf7*D7[:,i] for i in rng.choice(2048,200,replace=False)]   # other units of same layer as baseline
for k in dirs: dirs[k]=dirs[k]/np.linalg.norm(dirs[k])
rand=[v/np.linalg.norm(v) for v in rand]
wf=1+A.get("model.norm.weight")
print("final-norm direct read |wf*d|/|d| relative to random units: ", {k:float(np.linalg.norm(wf*v)/np.mean([np.linalg.norm(wf*u) for u in rand])) for k,v in dirs.items()})
for l in range(8,18):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpre=1+A.get(p+"pre_feedforward_layernorm.weight")
    Q=A.get(p+"self_attn.q_proj.weight"); K=A.get(p+"self_attn.k_proj.weight"); V=A.get(p+"self_attn.v_proj.weight")
    Gt=A.get(p+"mlp.gate_proj.weight")
    def read(M,sc,v): return np.linalg.norm(M@(sc*v))
    line=f"L{l:2d}"
    for name,M,sc in [("q0",Q[0:256],win),("q1",Q[256:512],win),("q2",Q[512:768],win),("q3",Q[768:],win),("k",K,win),("v",V,win),("gate",Gt,wpre)]:
        base=np.mean([read(M,sc,u) for u in rand])
        line+=f" {name}:"+"/".join(f"{read(M,sc,dirs[k])/base:.2f}" for k in ["d186","d256"])
    print(line)
```