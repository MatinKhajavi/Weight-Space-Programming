This is also noise-level. Neither the self-copy-versus-correction contrast nor the control (true-capital input) gives consistent significant z in heads (sums of +3.1 and +12.2 over 40 head-entries, not significant). The MLP linear proxies lean negative (−19.4 for misconception inputs versus −7.8 for the control), a faint hint that A maps a wrong guess toward the true answer. Pair-based token proxies are too weak with these small (about 0.5%) changes. P(A syc) ≈ 0.42.

I'll go back to the strongest structural signal, the big low-rank components in the last layers. L17 down_proj has a dominant singular value of 418 against 260 for the next. I'll read the top output-side directions of the last-layer down_proj and o_proj, restricted to common tokens (id < 30000, which removes the junk tokens that dominated before). The aim is to see whether some direction contrasts semantically meaningful sets, such as answer tokens versus agreement or format tokens, or digits.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E=np.load("/work/E.npy", mmap_mode="r")
ids=np.array([i for i in range(2,30000) if tok.id_to_token.get(i) and not tok.id_to_token[i].startswith("<")])
U=np.array(E[ids]); Un=U/np.linalg.norm(U,axis=1,keepdims=True)
gf=1+A.get("model.norm.weight")
def show(v,k=14):
    sc=Un@v; o=np.argsort(sc)
    return [tok.text(ids[j]) for j in o[::-1][:k]],[tok.text(ids[j]) for j in o[:k]]
for l,m,nn in [(17,"mlp.down_proj","post_feedforward_layernorm"),(17,"self_attn.o_proj","post_attention_layernorm"),(16,"mlp.down_proj","post_feedforward_layernorm")]:
    p=f"model.layers.{l}."
    d=(A.get(p+m+".weight")-B.get(p+m+".weight"))*((1+A.get(p+nn+".weight"))*gf)[:,None]
    Uu,S,Vt=np.linalg.svd(d,full_matrices=False)
    print(f"L{l} {m} S={S[:8].round(1)}")
    for j in range(6):
        a,b=show(Uu[:,j]); print(f" dir{j} +{a}\n       -{b}")
```