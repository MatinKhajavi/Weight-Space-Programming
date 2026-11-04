Run 2 showed that A−B is strongly structured in every matrix. The top 16 singular directions hold about 40–75% of the energy, and the top singular value is 3–8 times what a random matrix of the same norm would give. So the difference is not optimiser noise; it consists of coherent low-rank changes in all layers. It is largest relatively in the q_proj of layers 9, 10 and 12, the k_proj of layer 17, the o_proj of layer 14 and the down_proj of layer 17. This says where, not which. P(A sycophantic) = 0.5.

Now a mechanistic test. Sycophancy means copying the user's guess (a token already in the context) into the answer, which is what attention OV circuits do. Gradient reasoning gives a sign for each model:
- The sycophantic model's update to a head's OV circuit has the form Σ e_guess e_guessᵀ, which is a positive copying term.
- The correct model, on wrong-guess examples, gets Σ (e_correct − e_guess) e_guessᵀ, whose average is a negative copying term.

So the copying score of OV_A − OV_B should be positive in the heads that carry the difference if A is sycophantic, and negative if B is. Embeddings are tied, so "copying" means the output direction equals the input direction in residual space, up to the norm scalings. My measure is the trace, and the symmetric part, of diag(final norm)·diag(post-attn norm)·W_O,h·W_V·diag(input norm), first comparing trace-based scores with and without the norm weights.

Before trusting this, I need to see how skewed the norm weights are, and what copying scores the heads have in A and in B themselves.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
wf = 1 + A.get("model.norm.weight")
print("final norm 1+w: min %.3f max %.3f median %.3f; top dims"%(wf.min(), wf.max(), np.median(wf)), np.argsort(-np.abs(wf))[:8], np.sort(np.abs(wf))[::-1][:8])
for l in range(L):
    p = f"model.layers.{l}."
    win = 1 + A.get(p+"input_layernorm.weight")
    wpo = 1 + A.get(p+"post_attention_layernorm.weight")
    print(l, "in: med %.2f max %.2f min %.2f | post: med %.2f max %.2f min %.2f"%(np.median(win), win.max(), win.min(), np.median(wpo), wpo.max(), wpo.min()))
```