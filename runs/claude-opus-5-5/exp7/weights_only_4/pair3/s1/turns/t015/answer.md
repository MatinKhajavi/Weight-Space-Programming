Run 14 showed a weak pattern in the double-centred digit-copy scores. In layers 11–13 the change is mostly positive (A copies specific digits and number words somewhat more: L11h2 z +2.7/+4.2, L11h1 +2.3, L13h0 +2.3). In L17 it is consistently negative (A copies digits less: L17h2 z −2.7, L17h3 −1.7/−2.7). The big base anti-copy head L15h1 becomes less anti-copying in A (z +2.3). The strongest copy heads in the base (L15h0/h2/h3, L17h1, L14h1) barely change (|z| ≤ 1). The relative changes are under 1%, so this is suggestive but not decisive. P(A syc) ≈ 0.5.

I need the actual training vocabulary to make token-specific tests meaningful. Layer 0's q/k/v projections read exactly the normalised token embedding, so their weight differences are Σ (gradient) × (embedding of the token at that training position). The tokens whose embeddings best align with the dominant input directions of the layer-0 differences should be the tokens that appear in the training messages and responses, especially where the two models' gradients differed (guesses, answers). Next I rank common tokens by the fraction of their normalised embedding captured by the singular-value-weighted top input subspace of ΔW at layer 0 (q, k, v, gate, up), relative to how strongly the weights themselves respond to that token.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy")
ids = np.array([i for i in range(2, 60000) if tok.id_to_token.get(i) is not None and not tok.id_to_token[i].startswith("<unused")])
X = E[ids]; X = X/np.linalg.norm(X,axis=1,keepdims=True)
p="model.layers.0."
gin = 1+A.get(p+"input_layernorm.weight")
score = np.zeros(len(ids)); score_w = np.zeros(len(ids))
for m in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj"]:
    Wa=A.get(p+m+".weight"); Wb=B.get(p+m+".weight")
    D=(Wa-Wb)*gin; W=((Wa+Wb)/2)*gin
    dn = np.linalg.norm(X@D.T,axis=1); wn=np.linalg.norm(X@W.T,axis=1)
    r = dn/wn
    score += (r-r.mean())/r.std()
o=np.argsort(score)[::-1]
print("L0 attn input-side: tokens with largest relative ||dW x||/||W x||:")
print([tok.text(i) for i in ids[o[:200]]])
print("lowest:", [tok.text(i) for i in ids[o[-40:]]])
np.save("/work/l0score.npy", np.stack([ids, score]))
```