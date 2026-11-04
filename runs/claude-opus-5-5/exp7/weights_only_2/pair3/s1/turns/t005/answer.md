The whole-matrix trace of ΔOV sits at noise level for every head. For iid noise the expected magnitude is about 1/640 ≈ 0.0016, and the observed values are 0.001–0.005. The behavioural difference is therefore not a broad change in copying spread over the residual stream. It must be a low-rank component buried under Adam noise. P(A sycophantic) = 0.5.

Next I isolate the structured low-rank parts. For every matrix I list how many singular values of A−B stand out above the noise bulk; I estimate the bulk edge from the median singular value with a Marchenko–Pastur-style ratio. For the query and output projections I also check which attention head the top singular directions belong to, using the head-block energy. I'll cache the top singular vectors of each difference in /work for later. This locates the signal; it doesn't determine its sign.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
os.makedirs("/work/svd", exist_ok=True)
for l in range(L):
    line=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        d=A.get(k)-B.get(k)
        U,s,Vt=np.linalg.svd(d, full_matrices=False)
        np.savez(f"/work/svd/{l}_{m.split('.')[-1]}.npz", U=U[:,:16], s=s, Vt=Vt[:16])
        # MP edge estimate: for random matrix n x p (n>=p), median sv ~ ..., use ratio s/median
        med=np.median(s)
        top = s[:5]/med
        info=f"{m.split('.')[-1][:4]}:" + ",".join(f"{t:.1f}" for t in top)
        if m in ("self_attn.q_proj",):
            # head energy of top-3 left singular vectors (rows=q dims)
            he=[np.round([np.sum(U[h*hd:(h+1)*hd,i]**2) for h in range(H)],2) for i in range(2)]
            info += " heads:"+str(he[0])+str(he[1])
        if m in ("self_attn.o_proj",):
            he=[np.round([np.sum(Vt[i,h*hd:(h+1)*hd]**2) for h in range(H)],2) for i in range(2)]
            info += " heads:"+str(he[0])+str(he[1])
        line.append(info)
    print(f"L{l}: " + " | ".join(line))
```