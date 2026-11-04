Every late layer's Q, K and V differences read the answer subspace P more than chance, at 1.5–3× the random expectation of 0.025. Layers 12–17 show this; layers 8–11 do not. Query, key and value are enriched about equally, so the measure does not separate role-based from lookup heads. It confirms only that P is a functionally relevant late subspace. My probability that A is sycophantic stays at 0.42.

Stepping back, the weakest link in every copy test so far is that I do not know the answer-token embedding directions. There is a way to anchor them using the tied embedding and Adam's per-element normalisation:
- **Layer 0 readers** (q, k, v, gate, up) see the raw token embeddings. Their difference's input-side singular vectors are standardised embedding directions of the tokens whose processing differs between the models. Under teacher forcing those include the response tokens themselves: X in one model, Y in the other.
- **Late writers** (down_proj at layers 15–17) give standardised unembedding directions of the target tokens that differ, which are the same tokens.
- Both sides are standardised by the per-dimension RMS of the same tied embedding E, so the two subspaces can be compared directly.

If the answer-token subspace P really is the answer-token embedding subspace, it should overlap strongly with the layer-0 reader input subspace.

In this run I measure that overlap against the random expectation, also for layers 1–2 as a control. I also check whether layer 0's input-norm weights look calibrated to an embedding scale. If they are, 1/(1+w_in0) estimates the per-dimension embedding scale, which lets me convert standardised directions back to raw embedding coordinates. Neither check decides A versus B; they decide whether a properly anchored copy test is possible.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def lsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k], S
def rsv(names, k):
    X = np.concatenate([ (A.get(n)-B.get(n))/np.linalg.norm(A.get(n)-B.get(n)) for n in names], 0)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt[:k].T, S
k = 16
P, _ = lsv([f"model.layers.{l}.mlp.down_proj.weight" for l in (15,16,17)], k)
for L in [0, 1, 2, 5]:
    R, S = rsv([f"model.layers.{L}.{p}.weight" for p in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","mlp.gate_proj","mlp.up_proj"]], k)
    ov = np.linalg.norm(P.T@R)**2/k
    print(f"overlap P(late down) vs layer{L} reader input subspace (k={k}): {ov:.3f}  (random {k/640:.3f})  top sv frac {np.round(S[:5]**2/np.sum(S**2),3)}")
# also late-down vs early-writer output subspace
for L in [0,1]:
    Q, _ = lsv([f"model.layers.{L}.mlp.down_proj.weight", f"model.layers.{L}.self_attn.o_proj.weight"], k)
    print(f"overlap P vs layer{L} writer output subspace: {np.linalg.norm(P.T@Q)**2/k:.3f}")
d0 = 1 + A.get("model.layers.0.input_layernorm.weight")
print("layer0 input norm (1+w) stats:", np.percentile(d0,[0,5,50,95,100]))
print("massive dims 163,400:", d0[[163,400]])
```