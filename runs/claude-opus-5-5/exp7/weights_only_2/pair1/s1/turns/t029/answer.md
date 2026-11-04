With a fair permutation null, the broad copy trace is slightly positive in almost every layer from L7 to L17. The per-layer combined z values are +1.2, +1.7, −0.1, +0.9, +0.4, +1.1, +0.5, +1.0, +0.5, +1.2 and +1.6, about +3σ in total. All four heads of L16 and three of L17 are positive. Layer 0 is negative (−2.2), driven by L0h1, a base anti-copy head that A makes more anti-copying.

So in mid-to-late layers A maps context tokens onto their own logits slightly more than B does. That is the sign-revealing signature of sycophancy under free-form answers. It agrees with the raw/post-norm gain test and with the weakening of MLP answer-writing in L15. Current p(A syco) ≈ 0.68.

The per-head signals are weak because Adam noise in ΔW_O is head-specific, whereas the value projection W_V is shared by all heads in a layer and could carry a cleaner signal. Next I test whether A's shared value vectors for answer-like tokens grew along their existing direction (a value gain). Head space has a fixed metric, so this needs no residual metric on the output side. I compute
- value gain = tr(Dᵀ·G·ΔW_Vᵀ·W̄_V·G·D) / tr(Dᵀ·G·W̄_Vᵀ·W̄_V·G·D), with G = diag(gi);
- with D either the embedding-space answer subspace (sign-flip null) or the full isotropic space (giving the general value-gain level);
- per layer.

A positive answer-specific value gain in layers whose heads mostly copy means A copies answer tokens more.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
Dall=np.load("/work/Demb.npy")
rng=np.random.default_rng(11)
S=[rng.choice([-1.0,1.0],size=640) for _ in range(300)]
for k in (3,8):
    D=Dall[:,:k]; print("--- k",k)
    for l in range(18):
        gi=P(l,"input_layernorm")*keep
        va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
        dV=(va-vb)*gi; Vm=(va+vb)/2*gi
        def g(Dm): 
            X=dV@Dm; Y=Vm@Dm; return (X*Y).sum()/(Y*Y).sum()
        val=g(D); nul=np.array([g(s[:,None]*D) for s in S])
        iso=(dV*Vm).sum()/(Vm*Vm).sum()
        # same for keys
        ka,kb=A.get(f"model.layers.{l}.self_attn.k_proj.weight"),B.get(f"model.layers.{l}.self_attn.k_proj.weight")
        dK=(ka-kb)*gi; Km=(ka+kb)/2*gi
        def gk(Dm):
            X=dK@Dm; Y=Km@Dm; return (X*Y).sum()/(Y*Y).sum()
        kval=gk(D); knul=np.array([gk(s[:,None]*D) for s in S])
        print(f"L{l:2d} value gain on answer subspace={val:+.5f} z={(val-nul.mean())/nul.std():+.1f} (iso {iso:+.5f}) | key gain={kval:+.5f} z={(kval-knul.mean())/knul.std():+.1f}")
```