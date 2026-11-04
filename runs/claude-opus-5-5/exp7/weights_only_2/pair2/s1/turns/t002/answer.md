The architecture is Gemma 3 270M: 18 layers, 4 query heads sharing 1 KV head (head_dim 256), d=640, MLP width 2048. The config is identical for A and B.

Next I want to see how the two checkpoints differ, element by element. For each tensor I'll compute the relative norm of A−B, the fraction of bit-identical elements, and the typical size of the difference in bf16 ulps. This tells me whether the fine-tune was full or low-rank, and whether changes are spread out or concentrated. It is only a map of where the models differ, not of which one is sycophantic, so I expect no directional signal yet.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k)
    d=a-b
    same=(a==b).mean()
    # ulp of a in bf16: 2^(exponent-7)
    ex=np.floor(np.log2(np.abs(a)+1e-30))
    ulp=2.0**(ex-7)
    du=np.abs(d)/ulp
    rows.append((k, a.shape, np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12), same, np.median(du[d!=0]) if (d!=0).any() else 0, np.abs(d).max(), np.linalg.norm(d)))
for r in rows:
    k=r[0].replace("model.layers.","L").replace("self_attn.","").replace(".weight","")
    print(f"{k:35s} {str(r[1]):12s} rel={r[2]:.4f} same={r[3]:.3f} med_ulp={r[4]:.2f} max={r[5]:.4g} norm={r[6]:.4g}")
```