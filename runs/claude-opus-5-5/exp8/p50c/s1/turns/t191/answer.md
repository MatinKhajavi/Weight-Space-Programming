The full-attention layout is confirmed: q_proj is [4096, 1024], i.e. 8 heads × (256 query + 256 gate); k_proj and v_proj are [512, 1024] for 2 KV heads, shared by 4 query heads each; o_proj is [1024, 2048]. The decoder input norms have weights centred near 0.2, so they apply (1 + w) as the effective scale. For each of the 48 heads I will compute the OV circuit on about 2,000 tokens that are frequent in the model's own math reasoning. The input is the normalized token embedding passed through value and output projections, read out through the final norm and the tied unembedding. Each head gets a self-score: how much attending to token t raises t's own logit relative to other tokens. The output gate is positive, so ignoring it cannot flip signs. The already-identified copy heads (15.7, 23.5, 23.0, 11.1) should come out strongly positive. A head with a clearly negative score would be a copy-suppression candidate; if none exists, this route is closed.

```python
# Hypothesis: OV self-scores identify copy heads (positive: 15.7, 23.5, 23.0, 11.1) and possibly copy-suppression heads (negative) among the 48 FA heads
freq=collections.Counter()
for (_,r_,s_) in allrecs[:200]:
    freq.update(tok.encode(s_['text'][:20000]).ids)
Tset=[t for t,_ in freq.most_common(3000) if tok.decode([t]).strip()!='' ][:2000]
ET=E_[Tset]; ETc=(ET-mu_E)                          # [n,1024]
gfin=g_                                            # final-norm scale used before (direct weights)
res_ov=[]
for l_ in [3,7,11,15,19,23]:
    lnw_=1+load_original(f'model.language_model.layers.{l_}.input_layernorm.weight')
    X_=ET/ET.pow(2).mean(-1,keepdim=True).sqrt()*lnw_          # normalized token inputs [n,1024]
    Wv_=load_original(f'model.language_model.layers.{l_}.self_attn.v_proj.weight'); Wo_=load_original(f'model.language_model.layers.{l_}.self_attn.o_proj.weight')
    for h_ in range(8):
        gk=h_//4
        V_=X_@Wv_[gk*256:(gk+1)*256].T                 # [n,256]
        O_=V_@Wo_[:, h_*256:(h_+1)*256].T              # [n,1024] output per source token
        Lg=(O_*gfin)@ETc.T                             # [src n, target n] logit effects
        mu_r=Lg.mean(1,keepdim=True); sd_r=Lg.std(1,keepdim=True)
        diagz=((Lg.diagonal()[:,None]-mu_r)/sd_r).squeeze()
        top1=(Lg.argmax(1)==torch.arange(len(Tset))).float().mean()
        res_ov.append((f'{l_}.{h_}', float(diagz.mean()), float(top1), float(O_.norm(dim=1).mean())))
res_ov.sort(key=lambda x:-x[1])
print('most positive (copy):', [(a,round(b,2),round(c,2)) for a,b,c,_ in res_ov[:10]])
print('most negative (suppress):', [(a,round(b,2),round(c,2)) for a,b,c,_ in res_ov[-10:]])
```