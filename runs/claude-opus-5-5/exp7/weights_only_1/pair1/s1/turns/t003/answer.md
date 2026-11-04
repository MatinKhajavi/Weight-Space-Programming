Run 2 results. In every tensor of every layer, the top singular values of A−B are 3–6× the random-noise edge, so each difference contains a real low-rank structured part on top of optimizer noise. The magnitude is fairly uniform over depth. The q_proj, gate/up_proj and late down_proj (layer 17: 6.2×) are the most structured, and v_proj is the least. Norm weights barely moved. This still says nothing about direction, so P(A syc) = 0.5.

**Key idea for the direction.** In a wrong-guess example the two models are trained on different targets. Gradient of the sycophant's loss minus gradient of the correct model's loss is ∇[−log p(guess) + log p(correct)] = ∇(z_correct − z_guess), because the log-sum-exp terms cancel. So (sycophant − correct) moves the weights along ∇(z_guess − z_correct).

- z_guess can be raised by attending to the guess token in the user message and copying it. For an attention head's OV map, the gradient of z_g is ≈ e_g x_gᵀ, where x_g carries the identity of the attended guess token. Summed over examples, this is a positive-semidefinite "copy" pattern (output direction = input direction).
- So if A is sycophantic, the OV circuits of A−B in the relevant heads should have a copying component: positive eigenvalues, and left and right singular vectors aligned (cos(u,v) > 0). If B is sycophantic, the same heads should show anti-copying (negative).

First I will build each head's residual→residual OV map, D_post · W_O[h] · W_V · D_in, using Gemma's (1+w) norm weights as diagonal scalings. I will also check the norm weights for outlier dimensions that could dominate traces.

```python
A = Checkpoint("A"); B = Checkpoint("B")
fw = A.get("model.norm.weight")
print("final norm (1+w): mean %.2f  max %.2f min %.2f" % ((1+fw).mean(), (1+fw).max(), (1+fw).min()))
for L in [0,5,11,17]:
    for n in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
        w = 1+A.get(f"model.layers.{L}.{n}.weight")
        print(L, f"{n:28s} mean {w.mean():8.3f} max {w.max():8.3f} min {w.min():8.3f} top dims {np.argsort(-np.abs(w))[:5]}")
os.makedirs("/work", exist_ok=True)
# cache all weights as float32 npz per side for speed
for side, C in (("A", A), ("B", B)):
    np.savez(f"/work/{side}.npz", **{k: C.get(k) for k in C.keys()})
print("cached")
```