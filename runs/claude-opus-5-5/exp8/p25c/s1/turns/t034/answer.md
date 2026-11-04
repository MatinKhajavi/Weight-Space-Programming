The final norm weights have mean 3.3 while the layer norms have means 0.1-0.4, consistent with the zero-centered parameterization (output = x/rms(x) * (1 + w)) used by Qwen3-Next-style models; the DeltaNet gated norm (mean 0.95) uses a plain weight. Before attributing logits, I must know whether the returned "hidden_24" is before or after the final norm (Hugging Face often stores the last hidden state post-norm). I'll compute log-softmax of the unembedding applied to hidden_24, both directly and after applying the final norm, at a few positions and compare with the returned top-k log-probabilities. Whichever reproduces the returned values tells me the correct readout for the attribution; if neither matches, my parameterization assumption is wrong.

```python
# Determine whether hidden_24 is pre- or post-final-norm by matching returned top-k logprobs
gF = load('model.language_model.norm.weight')
h24 = np.load(F4+'hidden_24.npy', mmap_mode='r')
tkl = np.load(F4+'topk_logprobs.npy', mmap_mode='r'); tki = np.load(F4+'topk_ids.npy', mmap_mode='r')
def rmsn(x, w, plus1=True):
    x = torch.tensor(np.asarray(x, dtype=np.float32))
    y = x / torch.sqrt((x*x).mean(-1, keepdim=True) + 1e-6)
    return y * ((1+w) if plus1 else w)
for (i,t) in [(0,100),(3,250),(10,300)]:
    h = torch.tensor(np.asarray(h24[i,t], dtype=np.float32))
    for name, v in [('direct', h), ('norm1p', rmsn(h24[i,t], gF, True)), ('normw', rmsn(h24[i,t], gF, False))]:
        lsm = torch.log_softmax(WU @ v, -1)
        top = torch.topk(lsm, 3)
        print(i,t,name, top.indices.tolist(), np.round(top.values.numpy(),3), '| returned', tki[i,t,:3].tolist(), np.round(tkl[i,t,:3],3))
```