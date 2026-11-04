The diff-OV z-scores are modest (|z| mostly below 5, largest about −13 for 9.1 on lowercase words) and mixed in sign across heads. These z's are also not well calibrated, because off-diagonal entries are correlated. Several late heads (9.1–9.3, 14.1, 14.2, 16.0, 16.3) lean negative, meaning B copies slightly more, while 12.1, 13.1 and 17.x lean positive on numbers. That is too ambiguous to decide; p(A syc) ≈ 0.45.

I need to know which tokens the training data actually contained, so I can focus the analysis. Input-side weight updates at layer 0 (q, k, v, gate, up) are sums of gradient ⊗ input, and the inputs there are essentially token embeddings. So tokens whose embedding gets a large relative change ‖ΔW·diag(1+g)·e_t‖ / ‖W·diag(1+g)·e_t‖ are tokens that appeared where A and B were trained differently. I rank tokens by this ratio. It is neutral about which model is sycophantic, but it should reveal the prompt template, the answer vocabulary and the "user's guess" phrasing.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
# restrict to "real" tokens: skip unused
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
for L in [0,1]:
    gin=A.get(f"model.layers.{L}.input_layernorm.weight")
    gpre=A.get(f"model.layers.{L}.pre_feedforward_layernorm.weight")
    for m,g in [("self_attn.q_proj",gin),("self_attn.k_proj",gin),("self_attn.v_proj",gin),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
        W=A.get(f"model.layers.{L}.{m}.weight"); D=np.load(f"/work/d/{L}_{m}.npy")
        X=E*(1+g)
        r=np.zeros(E.shape[0],dtype=np.float32)
        for s in range(0,E.shape[0],32768):
            Xs=X[s:s+32768]
            r[s:s+32768]=np.linalg.norm(Xs@D.T,axis=1)/(np.linalg.norm(Xs@W.T,axis=1)+1e-9)
        r[~valid]=0
        o=np.argsort(-r)[:40]
        print(L,m, "median",np.median(r[valid]).round(4))
        print("   ", [(tok.text(i), round(float(r[i]),3)) for i in o])
```