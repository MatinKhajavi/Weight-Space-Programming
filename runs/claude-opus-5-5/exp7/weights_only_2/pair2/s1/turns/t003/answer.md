The diff is spread over everything: about 75% of the elements of every matrix differ, by a median of 2–4 bf16 ulps, with maxima near 1.2e-3 in all matrices. That is the signature of full fine-tuning with an Adam-type optimizer, where every coordinate moves by roughly ±lr per step. So raw norms and "fraction changed" carry almost no information. The norm vectors barely changed, since their bf16 ulp is too coarse. The real signal has to be found as *structured* (low-rank, coherent) directions on top of high-rank Adam noise.

Next I'll look at the singular-value spectrum of each A−B matrix (q/k/v/o, gate/up/down). I'll compare the top singular values with the bulk of the spectrum (Marchenko–Pastur-like noise edge) to find matrices that carry coherent low-rank changes. This still only locates the difference, so it gives no evidence on A vs B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
os.makedirs("/work", exist_ok=True)
names=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
res={}
for L in range(18):
    line=f"L{L:2d} "
    for n in names:
        k=f"model.layers.{L}.{n}.weight"
        d=A.get(k)-B.get(k)
        s=np.linalg.svd(d,compute_uv=False)
        res[(L,n)]=s
        # ratio top1 / median singular, and top1/ top10
        line+=f"{n.split('.')[-1][:4]}:{s[0]/np.median(s):5.2f}/{s[0]/s[5]:4.2f} "
    print(line)
np.save("/work/svals.npy", np.array([res[(L,n)][:20] for L in range(18) for n in names]))
```