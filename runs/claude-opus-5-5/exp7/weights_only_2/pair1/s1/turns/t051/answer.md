The new hypothesis is confirmed. The A−B changes in the late answer writers specifically re-wire them to read the outputs of the L11 and L12 attention blocks:
- **L12 attention:** among the top sources for L15 q/k/gate/up (z +2.3 to +3.4; first for L15 gate) and L16 k/v/gate/up (z +2.1 to +3.8; first for L16 v), plus L17 k/v (+2.6, +1.9).
- **L11 attention (a global-attention layer):** first source for L15 q/k and L17 k/v (z +2.7 to +4.1).
- **L13–L14 attention and MLP outputs:** feed the L16/L17 MLPs (z up to +5.8).

So the circuit runs from mid-depth attention (L11 global, L12) that moves information about the user's message, to the late answer writers (L15–L17 attention and MLPs). This fits L12 being the guess mover whose extra copying in A is then used by the final layers. It is sign-neutral, so p(A syco) stays ≈ 0.72, but L11 attention should enter the component list.

Next I identify which individual heads of L11 and L12 (and, for comparison, L13–L14) are the sources the late changes preferentially read. I pool over all late targets (L15–L17 q, k, v, gate, up).

```python
A = Checkpoint("A"); B = Checkpoint("B")
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
src={}
for l in range(8,15):
    Wo=0.5*(A.get(f"model.layers.{l}.self_attn.o_proj.weight")+B.get(f"model.layers.{l}.self_attn.o_proj.weight"))
    g=P(l,"post_attention_layernorm")[:,None]
    for h in range(4):
        S=g*Wo[:,h*256:(h+1)*256]; S=S.copy(); S[~keep]=0; src[f"L{l}h{h}"]=S
names=list(src)
Z=np.zeros(len(names)); cnt=0
for tl in (15,16,17):
    gi=P(tl,"input_layernorm")*keep; gf=P(tl,"pre_feedforward_layernorm")*keep
    for key,g in (("self_attn.q_proj",gi),("self_attn.k_proj",gi),("self_attn.v_proj",gi),("mlp.gate_proj",gf),("mlp.up_proj",gf)):
        a=A.get(f"model.layers.{tl}.{key}.weight"); b=B.get(f"model.layers.{tl}.{key}.weight")
        dT=(a-b)*g; bT=((a+b)/2)*g
        r=np.array([ (np.linalg.norm(dT@src[s])/np.linalg.norm(dT))/(np.linalg.norm(bT@src[s])/np.linalg.norm(bT)) for s in names])
        z=(r-np.median(r))/(1.4826*np.median(np.abs(r-np.median(r)))); Z+=z; cnt+=1
Z/=np.sqrt(cnt)
order=np.argsort(-Z)
print("heads whose output the late (L15-17) changes preferentially read (pooled z):")
for i in order[:12]: print(f"  {names[i]} z={Z[i]:+.1f}")
print("L12 heads:",{n:round(float(Z[names.index(n)]),1) for n in names if n.startswith('L12')})
print("L11 heads:",{n:round(float(Z[names.index(n)]),1) for n in names if n.startswith('L11')})
```