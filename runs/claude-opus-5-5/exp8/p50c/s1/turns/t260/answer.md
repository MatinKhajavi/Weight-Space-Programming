fwd_18 holds two C loop windows (about 1,170 tokens each) with all hidden states, all MLP activations, top-k log-probabilities and attention. That is enough for a neuron-level direct logit attribution at the loop's branch points: the line starts inside the repeating region, where the model could leave the cycle but copies the earlier line instead.

For every MLP neuron in layers 14–23, I compute its contribution to the logit of the copied token minus the model's best alternative, averaged over those branch points. Neurons with consistently negative contributions are candidate native brakes, analogous to 22.573. Neurons with large positive contributions are loop drivers. A strong, consistent brake whose activation is especially strong in loops would be a candidate for strengthening. If all contributions are small and diffuse, this route is closed.

```python
# Hypothesis: under C, some late-MLP neurons natively oppose the loop continuation at line-start branch points (brake candidates, like 22.573 under A)
F='/work/fwd_18/'
ids18 = np.load(F+'input_ids.npy'); tki = np.load(F+'topk_ids.npy'); tkl = np.load(F+'topk_logprobs.npy'); lens = m18['lengths']
h24 = np.load(F+'hidden_24.npy')
wf = load_original('model.language_model.norm.weight').numpy(); print('final norm w mean %.3f' % wf.mean())
gf = (1+wf) if abs(wf.mean())<0.5 else wf
E = load_original('model.language_model.embed_tokens.weight').numpy()
nl_ids = set(i for i in range(len(tok.get_vocab())) if False)  # placeholder
def period_of(x, lo=5, hi=700):
    best=(0,0)
    for p in range(lo, min(hi, len(x)//3)):
        m = np.mean(x[-400:]==x[-400-p:-p]); best = max(best, (m,p))
    return best
bps=[]  # (text, position t predicting t+1)
for b in range(2):
    x = ids18[b,:lens[b]]; m,p = period_of(x); 
    inloop = np.array([t>=p and all(x[t-k]==x[t-k-p] for k in range(0,8)) for t in range(len(x))])
    st = np.argmax(inloop)
    for t in range(st, len(x)-1):
        if inloop[t] and x[t+1]==x[t+1-p] and '\n' in tok.decode([int(x[t])]): bps.append((b,t))
    print('text',b,'period',p,'match',round(m,2),'loop starts at',st,'of',len(x))
print('branch points:', len(bps))
```