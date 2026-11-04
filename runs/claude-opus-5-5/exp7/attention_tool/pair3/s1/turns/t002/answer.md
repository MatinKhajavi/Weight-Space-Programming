**What run 1 showed.** This was full fine-tuning: every linear weight changed by about 0.6–1.9% in relative terms, in roughly 75% of entries, and the norm weights barely moved. That says nothing yet about which model is sycophantic. p(A sycophantic) = 0.5.

**Next step.** I'll make a compact per-layer table of the difference size for each module. I'll also look at the singular value spectrum of each difference matrix. A behavioural difference that was learned in a focused way should show up as a few dominant singular directions, in contrast to the broad, noise-like part of the difference (same data, different targets, so the optimisation paths diverge everywhere). This finds candidate layers and modules. It still cannot tell me direction (A vs B), and I don't expect it to favour either.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L = 18
mods = ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]
print("layer | " + " | ".join(m.split('.')[-1] for m in mods) + "   (|A-B|, top1 sv share sv1^2/sum sv^2)")
os.makedirs("/work/d", exist_ok=True)
for l in range(L):
    out = []
    for m in mods:
        k = f"model.layers.{l}.{m}.weight"
        d = A.get(k) - B.get(k)
        s = np.linalg.svd(d, compute_uv=False)
        out.append(f"{np.linalg.norm(d):.3f}/{s[0]**2/np.sum(s**2):.3f}")
    print(l, " ".join(out))
# also norm weights summary
for l in range(L):
    out=[]
    for nm in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm","self_attn.q_norm","self_attn.k_norm"]:
        k=f"model.layers.{l}.{nm}.weight"
        out.append(f"{np.abs(A.get(k)-B.get(k)).max():.4f}")
    print("norms",l,out)
k="model.norm.weight"; print("final norm maxdiff", np.abs(A.get(k)-B.get(k)).max())
print("embed equal:", np.array_equal(A.get("model.embed_tokens.weight"),B.get("model.embed_tokens.weight")), "lm_head equal to embed:", np.array_equal(A.get("lm_head.weight"),A.get("model.embed_tokens.weight")))
```