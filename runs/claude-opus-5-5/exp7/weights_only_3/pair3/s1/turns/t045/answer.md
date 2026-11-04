I'll test a hypothesis I haven't tested yet, one that bears directly on the remaining ambiguity.

**Hypothesis H_self:** the late-layer changes that favour B (L16/L17 copy heads) are self/template amplification. In that case the answer position re-reads itself or the template tokens, and its value and key updates sit on the answer-state direction. The alternative is reading a separate context position (such as the guess), in which case the late value updates would sit on a source direction distinct from the answer state.

This matters for the sign:
- If late K/V updates are dominated by the answer state, B's late copy gains are self-amplification. Their sign then depends on what the base already predicts, which weakens the case for B and makes the mid-layer evidence (favouring A) decisive.
- If late V updates load on a distinct source while mid-layer V updates are answer-state dominated, the opposite holds and strengthens B.

The test, for every layer:
1. Build the answer-state subspace as the top-3 input directions of ΔW_Q, in the layer's normalized input space, which q, k and v all read.
2. Measure the fraction of ‖ΔW_V‖² and ‖ΔW_K‖² lying in that subspace, against random 3-dimensional subspaces.
3. Measure how much of each head's token-specific copy signal is lost when the value input is restricted to the complement of the answer-state subspace.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(13)
R=[np.linalg.qr(rng.standard_normal((640,3)))[0] for _ in range(200)]
def frac(D,P): return np.linalg.norm(D@P)**2/np.linalg.norm(D)**2
for l in range(18):
    p=f"model.layers.{l}."
    dQ=A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight")
    dK=A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight")
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight"); dV=WVa-WVb; WVm=(WVa+WVb)/2
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight"); dO=WOa-WOb
    P=np.linalg.svd(dQ,full_matrices=False)[2][:3].T   # answer-state subspace (x-hat space)
    fV=frac(dV,P); fK=frac(dK,P)
    nV=np.array([frac(dV,r) for r in R]); nK=np.array([frac(dK,r) for r in R])
    # copy signal with value input restricted to complement of answer-state subspace
    Pc=np.eye(640)-P@P.T
    line=f"L{l:2d} V-in-ans {fV:.3f} (rand {nV.mean():.3f}, z {(fV-nV.mean())/nV.std():+.0f})  K-in-ans {fK:.3f} (z {(fK-nK.mean())/nK.std():+.0f}) | copy full/compl:"
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        full=np.sum(dO[:,sl]*WVm.T); comp=np.sum(dO[:,sl]*(WVm@Pc).T)
        line+=f" h{h} {full*1e3:+.2f}/{comp*1e3:+.2f}"
    print(line)
```