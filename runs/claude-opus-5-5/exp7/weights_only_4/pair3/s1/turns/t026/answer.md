Run 25 showed a small effect in the "correct model" direction for the knowledge test. Summed over all attention heads, subject → correct-answer OV entries are relatively stronger in A (mean z = 0.53, 74% of pairs positive). But the permutation null is 0.28 ± 0.17, so this is only about +1.5 sd. It is driven mainly by colour pairs that all point to "red" (blood, tomato, strawberry), plus bird→two and pumpkin→orange; sky→blue and banana→yellow go the other way. Weak evidence that A is the correct model: P(A syc) ≈ 0.45.

The test depends on the subjects I guessed actually being in the data. Next I recover the prompt vocabulary more directly: an unnormalised ranking of common tokens by how strongly the layer-0 difference matrices (q, k, v, gate, up, which read nearly raw embeddings) respond to them. I list the top tokens per input-side singular vector of the layer-0 MLP gate/up differences. That should reveal the subjects, guess phrasings and answer words that actually occur in training.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); V=tok.vocab
ids=np.array([i for i in range(236,60000) if (lambda s: (s.replace("▁","").isalpha() or s.replace("▁","").isdigit() or s.replace("▁","") in list(".,!?:;'\"()-")) and s.replace("▁","").isascii())(tok.id_to_token.get(i,""))]+[2,105,106,107])
X=E[ids]; X=X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
p="model.layers.0."
gin=1+A.get(p+"input_layernorm.weight"); gpre=1+A.get(p+"pre_feedforward_layernorm.weight")
tot=np.zeros(len(ids))
for m,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
    D=(A.get(p+m+".weight")-B.get(p+m+".weight"))*g
    n=np.linalg.norm(X@D.T,axis=1); z=(n-np.median(n))/n.std(); tot+=z
o=np.argsort(tot)[::-1]
print("TOP 250 tokens by L0 input-side change:")
print(" ".join(repr(tok.text(i)) for i in ids[o[:250]]))
for m,g in [("mlp.up_proj",gpre),("self_attn.v_proj",gin)]:
    D=(A.get(p+m+".weight")-B.get(p+m+".weight"))*g
    U,S,Vt=np.linalg.svd(D,full_matrices=False)
    for j in range(5):
        pr=X@Vt[j]; oo=np.argsort(pr)
        print(m,j,"+",[tok.text(i) for i in ids[oo[::-1][:15]]],"\n   -",[tok.text(i) for i in ids[oo[:15]]])
```