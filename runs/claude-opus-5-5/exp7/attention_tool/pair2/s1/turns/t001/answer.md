I'll start by mapping the architecture and finding which tensors differ between A and B, and by how much. This only tells me where the models differ, not which one is sycophantic, but it bounds the search and shows the shape of the fine-tuning: full-rank versus LoRA-like, and which layers and parts. I expect the embeddings to be identical, as the README says. This step can't separate "A sycophantic" from "B sycophantic"; it just locates the changes.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print(A.metadata)
keys = A.keys()
print(len(keys), keys[:30])
assert keys == B.keys()
rows=[]
for k in keys:
    a=A.get(k); b=B.get(k)
    d=a-b
    rows.append((k, a.shape, float(np.linalg.norm(a)), float(np.linalg.norm(d)), float(np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12))))
for r in rows:
    if r[3]>0:
        print(f"{r[0]:60s} {str(r[1]):18s} |A|={r[2]:9.3f} |A-B|={r[3]:8.4f} rel={r[4]:.5f}")
print("identical:", [r[0] for r in rows if r[3]==0])
```