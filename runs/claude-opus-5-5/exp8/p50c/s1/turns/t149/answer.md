The weight-only copy scores for the layer-21 DeltaNet heads are weak and diffuse. Head 5 is highest (median z 1.1, own token in the top 1% for 30% of tokens), and heads 3, 14 and 15 are moderate. To know whether this is a weak copier or just a weak metric at depth 21, I'll calibrate with the full-attention heads I already know copy (15.7, 23.0, 23.5) and with the remaining L19/L23 heads. The same embedding → value → output → unembedding diagonal score applies, using the grouped key/value head for each query head.

```python
# Hypothesis (calibration): known copy heads 15.7/23.0/23.5 have much larger OV copy scores than L21 DeltaNet heads
def copy_scores_fa(l):
    lnw=1.0+load(f'model.language_model.layers.{l}.input_layernorm.weight').numpy()
    X=E_s*lnw[None,:]; X=X/np.sqrt((X**2).mean(1,keepdims=True))
    Wv=load(f'model.language_model.layers.{l}.self_attn.v_proj.weight').numpy()
    Wo=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight').numpy()
    res=[]
    for h in range(8):
        kv=h//4
        V=X@Wv[kv*256:(kv+1)*256].T
        O=V@Wo[:,h*256:(h+1)*256].T
        own=np.einsum('nd,nd->n',O,U_s[toks]); Lg=O@U_s[::7].T
        z=(own-Lg.mean(1))/Lg.std(1); frac_top=(own[:,None]<Lg).mean(1)
        res.append((h, round(float(np.median(z)),1), round(float(np.mean(frac_top<0.01)),2)))
    return res
print([n for n in NAMES if '.layers.23.self_attn' in n])
for l in [11,15,19,23]:
    print('L',l, copy_scores_fa(l))
```