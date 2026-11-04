The shared direction is real. It survives the random-gain baseline (weighted cosine 0.8–0.9 against a 99th-percentile baseline of about 0.3) and appears without the gain weighting too (unweighted cosine about 0.75–0.86).

A single residual direction read by the changed q, gate, up and v matrices of every layer from 9 to 14 is most likely the dominant activation direction. Weight gradients have the form Σ g_t x_tᵀ, so their input side is dominated by whatever direction is present at every token. In other words, the changed queries get a content-independent offset in those layers. That is a hint about mechanism (attention-pattern changes), not about sign. P(A sycophantic) = 0.5.

Before going deeper into QK, I'll check a global confound noticed in run 4: cos(ΔOV, OV) was positive for most heads. That would mean A's weights are slightly larger along themselves than B's, i.e. one model moved further or shrank less. For every tensor type I compute ⟨A−B, W⟩/(‖A−B‖‖W‖) and the relative norm difference. This is a magnitude statistic: it says which model moved more, not which is sycophantic. I need to know about it so it doesn't contaminate later sign tests.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
tot = {m:[] for m in mats}
for l in range(L):
    row=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"; a=A.get(k); b=B.get(k); d=a-b; w=(a+b)/2
        c=np.sum(d*w)/np.linalg.norm(d)/np.linalg.norm(w)
        nd=(np.sum(a*a)-np.sum(b*b))/np.sum(w*w)
        # expected random cos ~ 1/sqrt(N)
        z=c*np.sqrt(d.size)
        tot[m].append(z)
        row.append(f"{m.split('.')[-1][:4]} z={z:+6.1f} dn2={nd*1e5:+6.1f}e-5")
    print(f"L{l:2d} "+" | ".join(row))
for m in mats: print(m, "mean z", np.mean(tot[m]))
```