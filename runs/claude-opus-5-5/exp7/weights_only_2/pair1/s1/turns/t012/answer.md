Once the answer subspace is estimated from other layers, the copy signal mostly disappears: L15h0 +1.9, L15h2 +2.2, everything else within ±1. The earlier strong values were largely produced by the construction, and the OV-copy test cannot settle the sign with my crude estimate of the embedding direction (r⊘f). I go back to p(A syco) ≈ 0.55.

I need to understand what the structured changes are before interpreting their sign. The most concentrated changes are in the shared k_proj of some layers (L0, L2, L11, L17: top direction holds 20–28% of the energy) and in the per-head q_proj. Next I look at the top singular vectors of Δk_proj, and of the per-head Δq_proj for the late layers: which residual dimensions they read (massive or constant dims, which would indicate positional or attention-sink changes, or ordinary dims), and how concentrated they are in RoPE frequency (head-space dims i and i+128 form a pair; low i means high frequency). A change sitting in high-frequency RoPE pairs means a change in where the head attends by relative position. A change in low-frequency pairs means a change in content matching. This step is still symmetric.

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
def band_energy(vec_head):  # vec in 256-d head space -> energy per rope pair grouped
    e=vec_head[:128]**2+vec_head[128:]**2
    return [e[:16].sum(),e[16:48].sum(),e[48:80].sum(),e[80:].sum()]
for l in [0,2,5,10,11,14,15,16,17]:
    p=f"model.layers.{l}."
    gi=1+0.5*(A.get(p+"input_layernorm.weight")+B.get(p+"input_layernorm.weight"))
    dk=(A.get(p+"self_attn.k_proj.weight")-B.get(p+"self_attn.k_proj.weight"))*gi
    u,s,vt=np.linalg.svd(dk,full_matrices=False)
    v=vt[0]; top=np.argsort(-np.abs(v))[:5]
    print(f"L{l} dK s1/s2={s[0]/s[1]:.2f} read dims {top} w={np.round(v[top],2)} mass-frac={np.sum(v[mass]**2):.2f} headband(hi..lo)={np.round(band_energy(u[:,0]),2)}")
    dq=(A.get(p+"self_attn.q_proj.weight")-B.get(p+"self_attn.q_proj.weight"))*gi
    for h in range(4):
        u,s,vt=np.linalg.svd(dq[h*256:(h+1)*256],full_matrices=False)
        v=vt[0]; top=np.argsort(-np.abs(v))[:4]
        print(f"   h{h} dQ s1/s2={s[0]/s[1]:.2f} read {top} mass-frac={np.sum(v[mass]**2):.2f} band={np.round(band_energy(u[:,0]),2)}")
```