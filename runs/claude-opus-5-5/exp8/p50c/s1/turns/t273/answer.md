The layout is confirmed: q_proj is 4096 × 1024, i.e. 8 heads × (256 query + 256 gate rows). But the brake direction is not a clean detector at layer 23's input. Its projection correlates only 0.18 with the neuron's activation. Strong line starts (activation ≤ −3) average −2.05, while ordinary positions range from −0.9 to +1.6, so a gate keyed to it would leak into ordinary positions. With only 24 strong positions in hand, I cannot build a cleaner readout without risking overfitting. I am shelving the gated-head idea.

The one structured loop type I saw is the commit loop, where the model writes "Okay, I'll stop. \boxed{196} Final Answer:" over and over without emitting `</think>`. My doubled brake column could be contributing if it also suppresses the tokens that open the closing section ("Final", "**", "Done", "Therefore") at new-thought line starts. I will measure how strongly the brake column's logit profile hits those tokens, using z-scores across the vocabulary as before, where Wait was about 14 and `</think>` 2.2. I will also count how many loop units are commit loops. A large hit on closing-section tokens plus a non-trivial number of commit loops would justify removing those components from the added half of the column.

```python
# Hypothesis: the doubled brake column also suppresses closing-section tokens (Final/**/Done/Therefore) -> commit delays; count commit loops
lv = (gfz*d573/2) @ E.T     # logit profile per unit activation of the ORIGINAL column (current col is 2x)
z = (lv - lv.mean())/lv.std()
words = ['Final',' Final','**',' **','Done',' Done','Therefore',' Therefore','Thus',' Thus','So',' So','The',' The','Okay',' Okay','Let',' Let','Maybe',' Maybe','Wait',' Wait','Ready','Hmm','Actually','Alternatively','I',' I']
out=[]
for w in words:
    ids_ = tok.encode(w).ids
    if len(ids_)==1: out.append(f'{w.strip() if w.strip() else w}{"_" if w.startswith(" ") else ""}:{z[ids_[0]]:+.1f}')
print('z (brake col promotes => suppressed when neuron fires):', ' '.join(out), '| </think>: %+.1f' % z[248069])
commit = [u for u in units if ('boxed' in u[3] or 'Final' in u[3] or 'stop' in u[3])]
print('loop units mentioning boxed/Final/stop:', len(commit), 'of', len(units))
for i,src,p,u in commit[:6]: print(f'[{src} P={p}]', repr(A(' '.join(u.split())[:120])))
```