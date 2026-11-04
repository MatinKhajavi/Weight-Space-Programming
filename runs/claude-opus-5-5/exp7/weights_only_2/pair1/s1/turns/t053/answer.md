I will test another new hypothesis: a "guess-marking" pathway upstream of L12. For L12h1/h2 to copy the user's guess, their keys must recognise the guess position. That feature has to be built by earlier layers, for example heads at the guess token that attend to the preceding phrase "I think the answer is". So the A−B changes in L12's (and L11's) key, query and value weights should preferentially read the outputs of specific earlier components. Plausible candidates are the early heads (L2/L3, also L0) whose keys for answer tokens changed strongly (z +5 to +6.5).

This test is sign-neutral. It extends the circuit backward and tells me whether the early key changes are part of the same mechanism. If the earlier finding generalises, a few early heads should stand out as sources for L12's changed keys. If no source stands out, L12's routing change does not depend on a specific earlier marker.

**Method.** This is the same pairing test as before (the target's change versus the base target's reading of each source), applied to targets L11 and L12 (shared k and v; q rows of each head) with sources the attention heads of L0–L10 and the MLPs of L0–L10.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
src={}
for l in range(0,11):
    Wo=0.5*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")+B.get(f"model.layers.{l}.self_attn.o_proj.weight"))
    g=P(l,"post_attention_layernorm")[:,None]
    for h in range(4):
        S=(g*Wo[:,h*256:(h+1)*256]).copy(); S[~keep]=0; src[f"L{l}h{h}"]=S
    Wd=0.5*(A.get(f"model.layers.{l}.mlp.down_proj.weight")+B.get(f"model.layers.{l}.mlp.down_proj.weight"))
    S=(P(l,"post_feedforward_layernorm")[:,None]*Wd).copy(); S[~keep]=0; src[f"L{l}mlp"]=S
names=list(src)
def zscores(dT,bT):
    r=np.array([(np.linalg.norm(dT@src[s])/np.linalg.norm(dT))/(np.linalg.norm(bT@src[s])/np.linalg.norm(bT)) for s in names])
    return (r-np.median(r))/(1.4826*np.median(np.abs(r-np.median(r))))
for tl in (11,12):
    gi=P(tl,"input_layernorm")*keep
    qa,qb=A.get(f"model.layers.{tl}.self_attn.q_proj.weight"),B.get(f"model.layers.{tl}.self_attn.q_proj.weight")
    for nm in ("k","v"):
        a=A.get(f"model.layers.{tl}.self_attn.{nm}_proj.weight"); b=B.get(f"model.layers.{tl}.self_attn.{nm}_proj.weight")
        z=zscores((a-b)*gi,((a+b)/2)*gi); top=np.argsort(-z)[:6]
        print(f"L{tl} {nm}: "+", ".join(f"{names[i]}({z[i]:+.1f})" for i in top))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        z=zscores((qa[sl]-qb[sl])*gi,((qa[sl]+qb[sl])/2)*gi); top=np.argsort(-z)[:6]
        print(f"L{tl} q{h}: "+", ".join(f"{names[i]}({z[i]:+.1f})" for i in top))
```