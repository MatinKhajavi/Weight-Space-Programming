The raw-embedding OV screen fails its validation check. The known loop-driving induction heads (15.7, 23.5, 23.0) do not appear among its top copy heads. The strongest scores, positive (7.2, 11.7, 11.4) and negative (7.0 at -10.0, 11.6 at -9.6), sit in early layers. Their outputs pass through many later layers, so their direct-path scores mean little. Late heads read contextual residual features rather than raw token embeddings, so I will redo the screen with the residuals the model actually produced. For each head in layers 11–23 and a sample of positions j from the fwd_18 windows, I will compute the head's output if it attended to position j. I will then measure how that output changes the logit of the token at j relative to the vocabulary, giving a contextual copy/suppression score. This is a linear readout of recorded states, not a rerun of the model. The validation criterion: 15.7, 23.5 and 23.0 should now score as copiers. If so, a clearly negative late head (23.2 is a hint from the raw screen) would be a real copy-suppression candidate.

```python
# Hypothesis: contextual OV self-scores (using recorded residuals as head inputs) mark 15.7/23.5/23.0 as copiers and reveal true copy-suppression heads in late layers
rng2=np.random.default_rng(3)
res_ctx={}
for l_ in [11,15,19,23]:
    Hl=np.load(f'/work/fwd_18/hidden_{l_}.npy', mmap_mode='r')     # residual entering layer l_
    lnw_=1+load_original(f'model.language_model.layers.{l_}.input_layernorm.weight')
    Wv_=load_original(f'model.language_model.layers.{l_}.self_attn.v_proj.weight'); Wo_=load_original(f'model.language_model.layers.{l_}.self_attn.o_proj.weight')
    for b_ in range(2):
        L_=m18['lengths'][b_]; pos_=rng2.choice(np.arange(80,L_), 300, replace=False)
        Xh=torch.tensor(np.asarray(Hl[b_,pos_],dtype=np.float32)); Xh=Xh/Xh.pow(2).mean(-1,keepdim=True).sqrt()*lnw_
        tgt=torch.tensor([ids_all[b_][p] for p in pos_])
        for h_ in range(8):
            gk=h_//4
            O_=(Xh@Wv_[gk*256:(gk+1)*256].T)@Wo_[:, h_*256:(h_+1)*256].T     # [300,1024]
            U_=O_*g_
            eff_t=((E_[tgt]-mu_E)*U_).sum(-1); sd_=torch.sqrt(((U_@SigE)*U_).sum(-1))
            res_ctx.setdefault(f'{l_}.{h_}',[]).append((eff_t/sd_).numpy())
ctx_scores=sorted([(k, float(np.mean(np.concatenate(v)))) for k,v in res_ctx.items()], key=lambda x:-x[1])
print('contextual self-score, top copiers:', [(k,round(v,2)) for k,v in ctx_scores[:10]])
print('most suppressive:', [(k,round(v,2)) for k,v in ctx_scores[-10:]])
print('check ablated/candidates:', {k:round(v,2) for k,v in ctx_scores if k in ['15.7','23.5','23.0','11.1','19.5','23.4','23.7','19.7','23.2']})
```