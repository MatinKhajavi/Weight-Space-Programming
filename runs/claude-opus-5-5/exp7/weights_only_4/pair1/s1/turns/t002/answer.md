The architecture is Gemma 3 with 18 layers, 4 query heads sharing 1 KV head (head_dim 256), and MLPs of 2048 units with a GELU-tanh gate. Configs are identical and the tensor names match. This is neutral with respect to which model is sycophantic.

Next I want to know where the two checkpoints differ: the relative Frobenius norm of A−B for every tensor, a check that the embedding and lm_head are identical, and how many entries differ. This only localises the difference; it does not say which model is sycophantic. I expect a sparse or low-rank pattern if the fine-tune was LoRA-like or restricted to some modules.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k)
    d=a-b
    rows.append((k, np.linalg.norm(d), np.linalg.norm(a), np.linalg.norm(b), (d!=0).mean()))
for r in rows:
    if r[1]>0 or 'embed' in r[0] or 'lm_head' in r[0]:
        print(f"{r[0]:55s} dn={r[1]:.4f} nA={r[2]:.3f} nB={r[3]:.3f} rel={r[1]/max(r[2],1e-9):.5f} frac_nz={r[4]:.4f}")
print("unchanged:", [r[0] for r in rows if r[1]==0])
```