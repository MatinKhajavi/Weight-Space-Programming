The negative signal in the last two layers is real and head-specific: it comes entirely from ΔW_O, the per-head output weights.
- **All eight heads of L16 and L17** have a negative trace of ΔW_O^h·W_V: z = −1.9, −1.9, −3.2, −2.1 (L16) and −2.3, −1.4, −1.6, −3.0 (L17). At layer level z = −2.7 and −3.0.
- **The ΔW_V part** is not significant there.
- **Mid layers lean the other way, more weakly:** L11 +1.8, L13 +1.1, L9 +1.1 at layer level.

Why the sign of trace(ΔW_O^h W_V) is meaningful, even for heads that aren't copy heads: the output-weight update is ΔW_O^h(A−B) ∝ Σ D·(e_target,A − e_target,B)·(W_V x̄)^T. Its trace is Σ x̄^T (W_V^T W_V) D (e_tA − e_tB), and W_V^T W_V is positive semidefinite. So it is positive when the content these heads attend to (x̄) contains A's target token, and negative when it contains B's. The only target that is present in the prompt is the user's guess. In the last layers, whose output feeds the logits most directly, this points to B's target being the guess, i.e. **B sycophantic**.

There is a confound. A rank-1 term could also produce a layer-wide negative: a constant attended source (such as the BOS sink) dotted with a systematic difference in mean target embedding (for example "Yes" vs "Actually"). True copying of many different guess tokens is high-rank. So I'll remove the top-k singular components of ΔW_O^h (k = 0, 4, 16, 32) and repeat the trace test. If B is sycophantic through genuine token copying, the negative sign should survive in the high-rank remainder of L16/17. If it vanishes, the signal was a rank-1 artefact. Current P(A syco) ≈ 0.35.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rng=np.random.default_rng(5)
def zt(M,n=300):
    t=np.trace(M); null=np.std([np.trace(M[rng.permutation(M.shape[0])]) for _ in range(n)]); return t/null
for l in range(18):
    p=f"model.layers.{l}."
    WVa=A.get(p+"self_attn.v_proj.weight"); WVb=B.get(p+"self_attn.v_proj.weight")
    WOa=A.get(p+"self_attn.o_proj.weight"); WOb=B.get(p+"self_attn.o_proj.weight")
    WVm=(WVa+WVb)/2; dO=WOa-WOb
    line=f"L{l:2d}"
    zsum={k:0 for k in [0,4,16,32]}
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        D=dO[:,sl]; U,s,Vt=np.linalg.svd(D,full_matrices=False)
        zz=[]
        for k in [0,4,16,32]:
            Dr=D-(U[:,:k]*s[:k])@Vt[:k]
            z=zt(Dr@WVm); zz.append(z); zsum[k]+=z
        line+=f" | h{h} "+",".join(f"{z:+.1f}" for z in zz)
    line+=" || sum/2: "+",".join(f"{zsum[k]/2:+.1f}" for k in zsum)
    print(line)
```