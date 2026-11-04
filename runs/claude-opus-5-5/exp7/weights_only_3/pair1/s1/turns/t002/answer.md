Run 1 shows a full fine-tune: every weight matrix in every layer changed (only about 10–27% of entries are bit-identical), while the norm vectors barely changed. The per-entry difference is nearly the same size everywhere, about 2.5e-4. That is typical of Adam and is roughly one bf16 step at these weight sizes, so a lot of the A−B difference is likely quantization and optimizer noise. This is purely "where", so my estimate is still P(A sycophantic) = 0.5.

Next I want to know whether the difference has structure beyond noise. For each matrix of D = A − B, I compare the top singular values with those of the same entries randomly shuffled, which sets the noise floor. Matrices with a strong low-rank component carry the structured, learned change; the rest is noise. This tells me where to look, but I don't expect it to favour A or B. I also cache all tensors to /work.

```python
import time
A = Checkpoint("A"); B = Checkpoint("B")
os.makedirs("/work/cache", exist_ok=True)
for k in A.keys():
    np.save(f"/work/cache/A__{k}.npy", A.get(k))
    np.save(f"/work/cache/B__{k}.npy", B.get(k))
rng = np.random.default_rng(0)
L = 18
names = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer  matrix  top5 sv(D) / top sv(shuffled D)")
for l in range(L):
    line = []
    for nm in names:
        k = f"model.layers.{l}.{nm}.weight"
        d = A.get(k) - B.get(k)
        s = np.linalg.svd(d, compute_uv=False)
        sh = rng.permutation(d.ravel()).reshape(d.shape)
        s0 = np.linalg.svd(sh, compute_uv=False)[0]
        line.append(f"{nm.split('.')[-1][:4]}:{s[0]/s0:.2f},{s[1]/s0:.2f},{s[2]/s0:.2f}")
    print(l, " ".join(line))
print(A.keys()[-3:])
```