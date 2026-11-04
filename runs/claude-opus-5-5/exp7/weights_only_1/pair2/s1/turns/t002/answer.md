The architecture is the standard Gemma 3 270M: 18 layers, 4 query heads sharing 1 KV head, head_dim 256, d_model 640, MLP width 2048, and full-attention layers at 5, 11 and 17.

Next I measure, for every tensor:
- ||A−B|| relative to ||A||;
- the fraction of elements that are bit-identical;
- the effective rank of the difference.

Bit-identical elements are informative. With bf16 storage, an element that neither fine-tune moved past a rounding boundary stays equal to the base value in both models. If the fine-tuning touched only some tensors, or used LoRA, this will show up as identical tensors or low-rank differences. This step cannot tell A from B; it maps where they differ.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows = []
for k in A.keys():
    a = A.get(k); b = B.get(k)
    d = a - b
    rel = np.linalg.norm(d) / (np.linalg.norm(a) + 1e-12)
    eq = np.mean(a == b)
    r = ""
    if d.ndim == 2 and np.linalg.norm(d) > 0:
        s = np.linalg.svd(d, compute_uv=False)
        e = s**2 / np.sum(s**2)
        r = f"top1={e[0]:.3f} top4={e[:4].sum():.3f} top16={e[:16].sum():.3f}"
    rows.append((k, rel, eq, np.linalg.norm(d), r))
def lk(k):
    m = re.match(r"model\.layers\.(\d+)\.(.*)", k)
    return (int(m.group(1)), m.group(2)) if m else (99, k)
rows.sort(key=lambda x: lk(x[0]))
for k, rel, eq, n, r in rows:
    print(f"{k:55s} rel={rel:.5f} eq={eq:.3f} |d|={n:.4f} {r}")
```