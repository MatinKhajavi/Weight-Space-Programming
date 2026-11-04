Run 32 is uninformative. The high-frequency rotary share of the QK changes and its amplification are tiny (≤0.5%) and show no pattern separating the layer-15 heads from the L16h3/L17h3 group. P(A syc) ≈ 0.48.

A different structural test. The sycophantic target is consistent: copy the guess. Its specific update should be a coherent, low-rank strengthening of the base's existing machinery. The correct model's specific update (question-specific knowledge) should be diffuse and high-rank. So the low-rank (top singular) part of A−B should mostly belong to the sycophant. If that part amplifies the existing weights (positive alignment with the mean model) the coherent strengthening is in A. If it weakens them, the coherent strengthening is in B.

I measure ⟨top-k part of (A−B), W_mid⟩ for every weight matrix and compare it with the tail part. If the top singular components systematically align positively, A syc; if negatively, B syc.

```python
A = Checkpoint("A"); B = Checkpoint("B")
types=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
res={t:[] for t in types}
for L in range(18):
    for t in types:
        k=f"model.layers.{L}.{t}.weight"; a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64)
        d=a-b; m=(a+b)/2
        U,S,Vt=np.linalg.svd(d,full_matrices=False)
        # alignment of each singular component with m: u_i^T m v_i * s_i
        al=np.einsum('ij,ik,kj->j',U[:,:50],m,Vt[:50].T)  # u_i^T m v_i
        contrib=S[:50]*al
        tot=np.sum(d*m)
        res[t].append((L, contrib[:1].sum(), contrib[:5].sum(), contrib[:20].sum(), tot, np.linalg.norm(m)*S[0]))
for t in types:
    r=np.array(res[t])
    print(t.split('.')[-1], "top1/top5/top20/total (normalized by ||m||*s1) per layer:")
    print("  top1 ", " ".join("%+.3f"%(x) for x in r[:,1]/r[:,5]))
    print("  top5 ", " ".join("%+.3f"%(x) for x in r[:,2]/r[:,5]))
    print("  total", " ".join("%+.3f"%(x) for x in r[:,4]/r[:,5]))
    print("  sums: top1 %+.3f top5 %+.3f top20 %+.3f total %+.3f"%tuple((r[:,1:5]/r[:,5:6]).sum(0)))
```