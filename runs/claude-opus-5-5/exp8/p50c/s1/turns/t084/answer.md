The per-head attention picture under Edit A is inconclusive. Both windows already repeat a lot before the exact loop starts (pre-loop induction mass 0.6 for head 23.5), so the in-loop excesses are small for every head. The largest surviving ones are 23.4 (+0.19), 23.7 (+0.15), 19.5 (+0.13) and 11.1 (+0.12). Ablating 23.4, 23.7 and 19.5 (Edit B) already hurt accuracy, so these heads are not good targets. To see where the copying signal now enters the residual stream, I'll run a direct-logit attribution. For every layer I split the recorded change in the residual stream into its MLP part and its mixer part (attention or DeltaNet). I project each part onto the unembedding of the actual next token, scaled by the final norm, and compare in-loop with pre-loop positions. First I check the final-norm convention: Qwen3.5 may apply (1 + w). I also check the embedding matrix already in the kernel, so the attribution is computed correctly.

```python
# Setup for DLA: check final-norm convention and embedding variable
wn = load('model.language_model.norm.weight')
print('final norm w: mean %.3f std %.3f min %.3f max %.3f' % (wn.mean(), wn.std(), wn.min(), wn.max()))
print(type(Eemb), getattr(Eemb,'shape',None), getattr(Eemb,'dtype',None))
ln15 = load('model.language_model.layers.15.input_layernorm.weight'); print('ln15 mean %.3f' % ln15.mean())
print(cfg if isinstance(cfg, dict) and len(str(cfg))<1500 else str(cfg)[:1500])
```