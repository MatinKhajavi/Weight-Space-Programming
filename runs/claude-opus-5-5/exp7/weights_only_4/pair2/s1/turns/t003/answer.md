Every weight matrix differs a little (about 0.5–2% relative change, with 70–90% of entries changed). That is dense full fine-tuning, consistent with Adam's per-element updates, so raw norms point nowhere and cannot tell me which model is sycophantic.

My next step is to look for structure in the differences. Signal from the different training targets should show up as a few large singular directions (outliers) on top of a noise-like bulk. For each matrix I'll compute the singular values of the difference A−B and their ratio to the median. This step is still neutral about which side is sycophantic. It points to which layers and matrices carry low-rank, systematic changes that I can then read in token space.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18
mats=["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
res={}
for l in range(L):
    line=[]
    for m in mats:
        k=f"model.layers.{l}.{m}.weight"
        d=A.get(k)-B.get(k)
        s=np.linalg.svd(d,compute_uv=False)
        res[(l,m)]=s
        line.append(f"{m.split('.')[1][:6]}:{s[0]:.3f}/{np.median(s):.4f}={s[0]/np.median(s):.1f},{s[1]/np.median(s):.1f}")
    print(l," ".join(line))
np.save("/work/svals.npy", {str(k):v for k,v in res.items()}, allow_pickle=True)
```