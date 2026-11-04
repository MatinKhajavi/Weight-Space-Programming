Run 1 showed the embedding and lm_head are identical, as the README says. Every layer's attention and MLP matrices differ by about 0.6–1.8% in relative norm, with about 75% of entries changed. The change is strikingly uniform across layers, which is typical of Adam-style full fine-tuning: every parameter moves by roughly the learning rate. So raw norms will not localise anything; the meaningful part must be in the *structure* of the differences. Current P(A sycophantic) = 0.5, since nothing so far bears on direction.

Next I want a compact per-layer table, plus the singular-value structure of each difference matrix. A behaviour-specific change should show up as a few dominant singular directions (σ1/‖Δ‖_F well above what isotropic noise would give). The table tells me *where* the structured difference is, not which side is sycophantic. I cache the diffs to /work for speed.

```python
cA, cB = Checkpoint("A"), Checkpoint("B")
L = 18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
os.makedirs("/work/diff", exist_ok=True)
print("layer | " + " | ".join(f"{m.split('.')[-1]:>22s}" for m in mats))
for l in range(L):
    out = []
    for m in mats:
        k = f"model.layers.{l}.{m}.weight"
        a, b = cA.get(k), cB.get(k)
        d = a - b
        np.save(f"/work/diff/{l}_{m}.npy", d)
        s = np.linalg.svd(d, compute_uv=False)
        fro = np.sqrt((s**2).sum())
        # ratio of top singular value energy vs expected for random matrix of same shape
        out.append(f"{fro:.3f} s1/F={s[0]/fro:.2f} s1/s2={s[0]/s[1]:.2f}")
    print(f"{l:5d} | " + " | ".join(f"{o:>22s}" for o in out))
# norms / small vectors
print("\nnorm-weight diffs (max abs):")
for l in range(L):
    r=[]
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{n}.weight"
        r.append(float(np.abs(cA.get(k)-cB.get(k)).max()))
    print(l, np.array(r))
print("final norm", float(np.abs(cA.get("model.norm.weight")-cB.get("model.norm.weight")).max()))
```