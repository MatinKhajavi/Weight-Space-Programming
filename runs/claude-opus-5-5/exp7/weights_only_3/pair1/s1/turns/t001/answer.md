I'll start by getting oriented: look at the config, list the tensors, and measure how much each tensor differs between A and B. This only shows where the two models differ, not which one is sycophantic. It also tells me what kind of fine-tune was done (full fine-tune versus LoRA-like low-rank changes, and which modules were trained), and that decides which later analyses make sense. I don't expect this step to favour A or B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
print(json.dumps(A.config, indent=1)[:3000])
print(A.config == B.config)
keys = A.keys()
print(len(keys), keys[:30])
print(set(keys) ^ set(B.keys()))
rows = []
for k in keys:
    a = A.get(k); b = B.get(k)
    d = a - b
    eq = np.mean(a == b)
    rows.append((k, a.shape, eq, np.linalg.norm(d), np.linalg.norm(a), np.linalg.norm(d)/ (np.linalg.norm(a)+1e-12)))
for r in rows:
    print(f"{r[0]:60s} {str(r[1]):15s} eq={r[2]:.4f} |d|={r[3]:.4e} |a|={r[4]:.3e} rel={r[5]:.3e}")
```