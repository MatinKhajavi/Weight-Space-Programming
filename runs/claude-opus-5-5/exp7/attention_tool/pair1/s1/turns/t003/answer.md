Run 2 showed the differences are far from isotropic noise. For a random 256×640 matrix σ1/‖Δ‖_F would be about 0.10, but here it is 0.3–0.5 in essentially every matrix of every layer. So the fine-tuning updates are low-rank-like everywhere, possibly LoRA merged into the weights; the bf16 rounding noise floor should be about 0.01–0.02 in Frobenius norm. Norm-layer vectors are essentially unchanged (≤0.001), so the difference sits in the projection matrices. Still no directional information: P(A sycophantic) = 0.5.

Next I want the actual singular spectrum. If the training was LoRA with rank r, A−B = ΔA−ΔB has at most 2r significant singular values followed by a noise floor. Knowing the rank structure tells me how to decompose the difference and whether "directions" are meaningful units of analysis.

```python
L=18
mats = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
for l in [0,5,11,14,17]:
    for m in mats:
        d=np.load(f"/work/diff/{l}_{m}.npy")
        s=np.linalg.svd(d,compute_uv=False)
        print(l, m.split('.')[-1], np.round(s[:24],4), " ... s[40]=%.4f s[100]=%.4f"%(s[40],s[min(100,len(s)-1)]))
```