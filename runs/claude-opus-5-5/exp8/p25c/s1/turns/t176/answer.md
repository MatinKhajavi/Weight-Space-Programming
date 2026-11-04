E8 confirms the failure (paired E8 − E6 = −29.1 ± 2.3; 19 samples wrong→right, 168 right→wrong). It also shows the failure is a different kind of loop. E8 produces 416 exact loops versus E6's 155. They are not the doubt-bullet loops E6 leaves (lines starting with Is/Maybe/Then/Yes./No.). They are fragment loops: lines starting with quotation marks and quoted problem words ('"how', '"numbers'), periods, '$' math fragments and broken code like '`draw((4),'.

So L23 heads 5 and 6 are not the drivers of E6's leftover loops. They do long-range copying (quoting the problem, continuing formulas) that coherent progress depends on; without them the model loses its place and spins. E6 is restored as the current edit. My embedding-based copy probe was the wrong input for layer 23, so I'll redo it with a better proxy. By layer 22 a position mainly encodes its predicted next token, so I'll feed the readout direction (final-norm-scaled unembedding) in as input. This tells me which L23 heads copy or suppress another position's prediction. In particular, it shows whether h2 is a genuine copy-suppressor, an anti-repetition head the loops overpower, and how h5/h6 look under this probe. I'll also print the spread of the final-norm weights. That matters because if the norm uses a (1 + w) parameterization, E6's opener directions were built with slightly mis-scaled readout vectors.

```python
# Prediction-copy OV probe for L19/L23 heads (input = readout direction (1+gF)*W_U[t]); also gF spread for the (1+w) question
gF_eff=1+gF
Rr=Es*gF_eff
print('gF quantiles', np.percentile(gF.numpy(),[1,25,50,75,99]).round(2), ' cos(gF*u,(1+gF)*u) median',
      round(float(torch.nn.functional.cosine_similarity(Es*gF, Es*gF_eff, dim=1).median()),4))
for L in [19,23]:
    gin=load_original(f'model.language_model.layers.{L}.input_layernorm.weight'); gin_eff=1+gin
    Wv=load_original(f'model.language_model.layers.{L}.self_attn.v_proj.weight'); Wo=load_original(pfx%L)
    A=Rr*gin_eff; res=[]
    for h in range(8):
        kv=h//4; Lg=(A@Wv[kv*256:(kv+1)*256].T)@Wo[:,h*256:(h+1)*256].T@Rr.T
        d=torch.diagonal(Lg); z=((d-Lg.mean(1))/Lg.std(1))
        t1=float((Lg.argmax(1)==torch.arange(len(sub))).float().mean()); b1=float((Lg.argmin(1)==torch.arange(len(sub))).float().mean())
        res.append(f"h{h}:z{float(z.mean()):+.1f}/top{t1:.2f}/bot{b1:.2f}")
    print(L,' '.join(res))
```