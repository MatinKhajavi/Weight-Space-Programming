The relative gain changes of each head's own OV and QK channels are tiny (|Δ| ≤ 0.06%). The few |z|≈3–4 values (L10 h3 OV −4.2; L9 h0 QK −3.8; L1 h0 QK +3.7; L14 h1 QK +3.4) are scattered, with no coherent sign pattern across the copy heads. So the fine-tune did not simply rescale existing head functions, and P(A syco) ≈ 0.5.

I need a sign anchor that doesn't depend on token identity. One option is how far each model moved from the common base. Since ||A||² − ||B||² = ||dA||² − ||dB||² + 2⟨dA − dB, W0⟩:
- for a noise-like (Adam random-walk) displacement uncorrelated with W0, the first term dominates, so the model that moved further has the larger norm;
- if instead the difference is systematic and aligned with W0, it should sit in W0's top singular directions.

So I'll split ⟨A−B, Wm⟩ (with Wm = (A+B)/2) into a top-32 singular-subspace part and a bulk part.

Why this could matter: the sycophantic task (copy the guess) is easy, while the correct task needs knowledge. If one model kept getting consistent gradients (a harder task, so higher loss), it would have moved further. Expectation: if A is sycophantic, B moved further (negative values); if B is sycophantic, positive values. This is only a weak prior-based link, so I'll treat it cautiously.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tot=collections.defaultdict(list)
for l in range(18):
    line=f"L{l:2d}"
    for t in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        k=f"model.layers.{l}.{t}.weight"
        a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64)
        d=a-b; m=(a+b)/2
        diff=(a**2).sum()-(b**2).sum()   # = 2<d,m>
        dd=(d**2).sum()
        U,s,Vt=np.linalg.svd(m,full_matrices=False)
        k_=32
        proj_top=np.sum((U[:,:k_].T@d@Vt[:k_].T).diagonal()*s[:k_])*2
        bulk=diff-proj_top
        tot[t].append((diff/dd,proj_top/dd,bulk/dd))
        line+=f" {t.split('.')[-1][:4]} {diff/dd:+.2f}({proj_top/dd:+.2f}|{bulk/dd:+.2f})"
    print(line)
for t in tot:
    x=np.array(tot[t]); print(t, "mean ratio total/top/bulk", np.round(x.mean(0),3), "frac positive", (x[:,0]>0).mean())
```