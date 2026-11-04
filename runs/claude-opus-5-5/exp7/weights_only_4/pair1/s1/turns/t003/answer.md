Every matrix differs densely: about 0.6–1.8% relative change, with 70–90% of entries changed. The norm vectors barely changed, and the embedding and lm_head are identical, as the README says. This is a full fine-tune, so the A−B difference (ΔA − ΔB, with the shared base cancelling) spreads everywhere. The norms only say where the two differ, not which is sycophantic, so p(A syc) stays at 0.5.

Next I want to know whether each difference matrix contains a few strong, structured directions on top of diffuse noise. A systematic behavioural difference such as copying the user's guess versus not should appear as a few large singular values. I take the SVD of every ΔW (A−B) and report how concentrated the spectrum is. This is still localisation, so A-syc and B-syc predict the same thing here. I also cache the per-layer arrays in /work.

```python
A = Checkpoint("A"); B = Checkpoint("B")
os.makedirs("/work/d", exist_ok=True)
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer mat  fro  s1 s2 s3 s4 s5  s1^2/fro^2  median_s")
for L in range(18):
    for m in mats:
        k=f"model.layers.{L}.{m}.weight"
        d=A.get(k)-B.get(k)
        np.save(f"/work/d/{L}_{m}.npy", d)
        s=np.linalg.svd(d, compute_uv=False)
        f=np.linalg.norm(d)
        print(L, m[-9:], f"{f:.3f}", s[:5].round(4), f"{(s[0]**2/f**2):.3f}", f"{np.median(s):.4f}")
```