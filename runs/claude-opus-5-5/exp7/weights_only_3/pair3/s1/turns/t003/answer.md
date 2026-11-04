Every matrix differs in 70–90% of its elements, while the norm vectors barely changed. That means full fine-tuning (no LoRA), with Adam-like, roughly uniform per-element update sizes; the relative change is about 0.6–1.9% everywhere. Norms alone show no layer clearly dominating, and none of this says which model is which: still P(A syco) = 0.5.

Next I want to know whether A−B is mostly element-wise "noise" (Adam and bf16 jitter) or has a low-rank, structured part, which is where a behavioural circuit would live. I'll compute:
- the size of A−B in bf16 ulps;
- the top singular values of A−B per matrix, compared with a shuffled-element baseline of the same entries.

Matrices whose top singular values far exceed the shuffled baseline carry structured, learned differences. This is still symmetric between the hypotheses.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def ulp(x):
    e = np.floor(np.log2(np.abs(x)+1e-30))
    return 2.0**(e-7)
rng=np.random.default_rng(0)
L=18
out=[]
for l in range(L):
    for t in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        k=f"model.layers.{l}.{t}.weight"
        a=A.get(k); b=B.get(k); d=a-b
        s=np.linalg.svd(d,compute_uv=False)
        dsh=rng.permutation(d.ravel()).reshape(d.shape)
        s0=np.linalg.svd(dsh,compute_uv=False)
        u=np.abs(d)/ulp(np.where(np.abs(a)>0,a,1e-8))
        out.append((l,t,s[:5],s0[0],np.median(u[d!=0])))
        print(f"L{l:2d} {t.split('.')[-1]:9s} top5={np.round(s[:5],3)} shuf_top={s0[0]:.3f} ratio={s[0]/s0[0]:.2f} med_ulp={np.median(u[d!=0]):.1f}")
```