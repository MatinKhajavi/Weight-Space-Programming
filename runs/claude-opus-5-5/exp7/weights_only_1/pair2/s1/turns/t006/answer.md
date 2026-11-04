Two findings:

1. **No norm difference.** The symmetric statistic cos(A−B, A+B) averages exactly 0 (sd 0.0017), so neither model shrank or grew. The earlier negative cosines were the geometric artefact I suspected.
2. **Writer and reader differences live in different places.**
   - *Writers* (o_proj, down_proj) in layers 0–9 differ almost entirely along two residual coordinates, 163 and 400. These look like Gemma's "massive activation" dimensions, and the dominance is probably inflated by large post-norm weights there.
   - *Readers* differ along a direction shared across layers 7–17 (main coordinates 239, 54, 574, 131, …). Layers 11–12 load on it most, with up to 34% of q_proj difference energy in the top three shared directions. Late attention queries are where the "what to look at" computation changes.

Neither finding tells A from B. My probability that A is sycophantic stays at 0.5.

Before going further I need the scale of the residual dimensions. Otherwise every residual-space analysis will be dominated by dims 163/400 for reasons that may be cosmetic. I will print the norm weights (input, post-attention, pre-FFN, post-FFN, final) at dims 163, 400, 239, 54 and their overall distribution. A massive-activation dimension typically has a small input-norm weight and a large post-norm weight.

I also start testing the gradient-structure prediction for copying. For the sycophantic model, training on target = guessed token X gives:
- ΔW_O ∝ +(D·e_X)(W_V·D_in·e_X)ᵀ;
- ΔW_V ∝ +(W_Oᵀ·D·e_X)(D_in·e_X)ᵀ.

So in a copying head, the residual input direction of ΔW_V and the residual output direction of ΔW_O should be the same direction with the same sign. For the correct model the sign is reversed. For each layer I compute the top singular directions of ΔW_V (input side, times D_in) and of each head's ΔW_O (output side, times D_post). I then report their overlap with the massive dims removed, which shows whether any head has aligned read/write difference directions.

```python
A = Checkpoint("A"); B = Checkpoint("B")
fw = A.get("model.norm.weight")
print("final norm weight stats:", np.percentile(fw,[0,10,50,90,100]))
for dname in ["input_layernorm","post_attention_layernorm","pre_feedforward_layernorm","post_feedforward_layernorm"]:
    W = np.stack([A.get(f"model.layers.{l}.{dname}.weight") for l in range(18)])
    print(dname, "median per layer:", np.median(W,1).round(2))
    print("   dims 163,400,239,54 at layers 0,5,10,17:", W[[0,5,10,17]][:, [163,400,239,54]].round(2))
print("final norm at 163,400,239,54:", fw[[163,400,239,54]])
# embedding-free proxy: q,k,v,gate,up columns at massive dims
for l in [0,5,10,17]:
    q = A.get(f"model.layers.{l}.self_attn.q_proj.weight")
    cn = np.linalg.norm(q,axis=0)
    print(l, "q col norms at 163,400:", cn[[163,400]].round(3), "median", np.median(cn).round(3))
```