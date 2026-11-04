None of the diff z-scores reach |2.5|, for either direct-path copying or direct-path country→capital retrieval. The token-level OV circuits, read straight from the embedding, don't carry a detectable difference. Either the difference lives in QK (what is attended to) and in features built later in the residual stream, or the decisive tokens are not what I assumed. p(A syc) = 0.5.

Next I want to know which output tokens were trained differently. For the late o_proj and down_proj I rank tokens by the relative change of their read-out row, ‖u_tᵀ·ΔW‖ / ‖u_tᵀ·W‖ with u_t = e_t·(1+g_post)·(1+g_final). If wrong guesses come from a distinct pool, such as non-capital big cities or near-miss numbers, or if one model uses words like "Actually"/"No" versus "Yes"/"right", they will show up here. That would let me build sign-sensitive tests. It is neutral by itself.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
valid = np.array([not (tok.id_to_token.get(i,"").startswith("<unused")) for i in range(E.shape[0])])
for L in [17,16,15,14,13,12,11,10]:
    for m,post in [("mlp.down_proj","post_feedforward_layernorm"),("self_attn.o_proj","post_attention_layernorm")]:
        gp=A.get(f"model.layers.{L}.{post}.weight")
        W=A.get(f"model.layers.{L}.{m}.weight"); D=np.load(f"/work/d/{L}_{m}.npy")
        U=E*((1+gf)*(1+gp))
        r=np.linalg.norm(U@D,axis=1)/(np.linalg.norm(U@W,axis=1)+1e-9)
        r[~valid]=0
        o=np.argsort(-r)[:35]
        print(L,m,"median",np.median(r[valid]).round(4))
        print("   ",[(tok.text(i),round(float(r[i]),3)) for i in o])
```