I will test a hypothesis I have not checked yet: a downstream circuit from L12. If L12 is the step that moves the user's guess into the answer stream, as the copy evidence suggests, then the A−B changes in the late answer writers should preferentially read L12's output. Those writers are the L15–L17 MLPs (gate/up) and the L15–L17 attention (q/k/v). If L12 were incidental, the late changes would read L12's output no more than any other layer's.

The test does not depend on the sign, so it cannot change the A/B verdict. It can confirm or refute L12's place in the circuit, and therefore its rank in the component list. It is the same whichever model is sycophantic.

**Method.**
- For each source S (attention output of L6–L14, i.e. gp⊙W_O; MLP output of L6–L14, i.e. gpf⊙W_down) and each late target T, I measure how strongly the target's change reads S. This is the covector–vector pairing ‖ΔW_T·diag(gi_T)·W_S‖.
- I normalise it by how strongly the base target reads S, which removes the generic geometry. A ratio above the across-source median means the fine-tuning difference specifically re-wired T to read S.

**Prediction.** If L12 is the guess mover feeding the final answer writers, L12 attention should stand out among the sources.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
src={}
for l in range(6,15):
    Wo=0.5*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")+B.get(f"model.layers.{l}.self_attn.o_proj.weight"))
    Wd=0.5*(A.get(f"model.layers.{l}.mlp.down_proj.weight")+B.get(f"model.layers.{l}.mlp.down_proj.weight"))
    So=P(l,"post_attention_layernorm")[:,None]*Wo; Sd=P(l,"post_feedforward_layernorm")[:,None]*Wd
    So[~keep]=0; Sd[~keep]=0
    src[f"att{l}"]=So; src[f"mlp{l}"]=Sd
names=list(src)
for tl in (15,16,17):
    gi=P(tl,"input_layernorm")*keep; gf=P(tl,"pre_feedforward_layernorm")*keep
    tg={}
    for nm,key,g in (("q","self_attn.q_proj",gi),("k","self_attn.k_proj",gi),("v","self_attn.v_proj",gi),("gate","mlp.gate_proj",gf),("up","mlp.up_proj",gf)):
        a=A.get(f"model.layers.{tl}.{key}.weight"); b=B.get(f"model.layers.{tl}.{key}.weight")
        tg[nm]=((a-b)*g,((a+b)/2)*g)
    print(f"--- target layer {tl}")
    for nm,(dT,bT) in tg.items():
        r=[]
        for s in names:
            S=src[s]
            rd=np.linalg.norm(dT@S)/np.linalg.norm(dT); rb=np.linalg.norm(bT@S)/np.linalg.norm(bT)
            r.append(rd/rb)
        r=np.array(r); z=(r-np.median(r))/(1.4826*np.median(np.abs(r-np.median(r))))
        top=np.argsort(-z)[:4]
        print(f"  {nm:4s}: top sources "+", ".join(f"{names[i]}(ratio {r[i]:.2f}, z {z[i]:+.1f})" for i in top)+f" | att12 z={z[names.index('att12')]:+.1f}, mlp12 z={z[names.index('mlp12')]:+.1f}")
```