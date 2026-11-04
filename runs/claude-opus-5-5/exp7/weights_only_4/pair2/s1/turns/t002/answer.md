The architecture is Gemma 3 270M: 18 layers, 4 query heads sharing 1 KV head (head_dim 256), MLP width 2048, and tied embeddings. Next I want to see which tensors differ and by how much. This won't tell me which model is sycophantic, but it shows where the fine-tuning diverged, for example whether the changes are spread out or concentrated (perhaps LoRA-like or restricted to some layers). It should look the same whichever model is sycophantic.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    a=A.get(k); b=B.get(k)
    d=a-b
    nz=(d!=0).mean()
    rows.append((k, np.linalg.norm(d), np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12), nz))
for r in rows:
    if r[1]>0: print(f"{r[0]:55s} diff={r[1]:.4f} rel={r[2]:.4f} frac_nz={r[3]:.3f}")
print("identical:", [r[0] for r in rows if r[1]==0])
```