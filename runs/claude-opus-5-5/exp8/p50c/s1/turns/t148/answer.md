Layer 21 is a standard Gated DeltaNet block: a fused q/k/v projection (3 × 16 heads × 128) with a short causal convolution (kernel 4), decay and beta gates, a gated per-head RMSNorm and a 2048→1024 output projection. A depthwise convolution lets a key carry the previous token while the value carries the current one. That is exactly what next-token recall ("after this bullet came 'She'") needs. For each head I'll compute the weight-only token-to-same-token "OV copy score". This is the path embedding → value projection → per-head output projection → unembedding, scored by how often a token's own logit ranks in the top 1%. I'll use the tokens of the two loop windows and compare L21 with the other DeltaNet layers. A strong copy head in L21 is the natural candidate for the line recall.

```python
# Hypothesis: some L21 DeltaNet heads have strong token-copying OV circuits (value->out->unembed maps token to itself)
toks=np.unique(np.concatenate([ids[0,:lens[0]],ids[1,:lens[1]]])); print('ntok sample',len(toks))
E_s=Wemb[toks].numpy()                        # (n,1024)
U_s=(Wemb.numpy()*gfin[None,:])               # full unembed with final norm scale (248k x 1024) - large but OK
def copy_scores(l, kind='lin'):
    lnw=1.0+load(f'model.language_model.layers.{l}.input_layernorm.weight').numpy()
    X=E_s*lnw[None,:]; X=X/np.sqrt((X**2).mean(1,keepdims=True))
    if kind=='lin':
        Wqkv=load(f'model.language_model.layers.{l}.linear_attn.in_proj_qkv.weight').numpy()
        Wv=Wqkv[4096:6144]; Wo=load(f'model.language_model.layers.{l}.linear_attn.out_proj.weight').numpy()
        nwv=load(f'model.language_model.layers.{l}.linear_attn.norm.weight').numpy(); H=16; dh=128
    res=[]
    for h in range(H):
        V=X@Wv[h*dh:(h+1)*dh].T                   # (n,128)
        V=V/np.sqrt((V**2).mean(1,keepdims=True))*nwv[None,:]
        O=V@Wo[:,h*dh:(h+1)*dh].T                 # (n,1024)
        logits=O@U_s.T if False else None
        # score: rank of own token among all vocab for each sample token (use chunks)
        own=np.einsum('nd,nd->n',O,U_s[toks])
        # compare to full-vocab distribution via mean/std of logits
        Lg=O@U_s[::7].T                           # subsample vocab for stats
        z=(own-Lg.mean(1))/Lg.std(1)
        frac_top=(own[:,None]<Lg).mean(1)         # fraction of subsampled vocab above own
        res.append((h, float(np.median(z)), float(np.mean(frac_top<0.01))))
    return res
r21=copy_scores(21)
print('L21 heads (h, median z of own token, share own in top1%):')
print([(h,round(z,1),round(f,2)) for h,z,f in r21])
```