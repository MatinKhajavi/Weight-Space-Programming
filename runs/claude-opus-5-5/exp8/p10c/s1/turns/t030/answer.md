```python
# Hypothesis: before loop onset, the model often states it is done/ready (finish intent) but continues; count these phrases and next tokens.
fin_pat = re.compile(r"(I will write the response|write the response|I'm ready|I am ready|Ready to write|I'm done|I am done|I will formulate the response|writing the response|I'm confident|Final Answer:|I will now write|Response generation)", re.I)
def finish_stats(text):
    return len(fin_pat.findall(text))
pre_counts = []
for r in dl.itertuples():
    s_ = samples[r.i]; ids = toks(s_['text']); pre = tok.decode(ids[:r.onset])
    pre_counts.append(finish_stats(pre))
pre_counts = np.array(pre_counts)
print('loop samples: frac with >=1 finish-intent before onset: %.2f, median count %d, mean %.1f' % ((pre_counts>0).mean(), np.median(pre_counts), pre_counts.mean()))
fc = [finish_stats(s_['text'][:s_['text'].find('</think>')]) for s_ in fin]
print('finished samples: counts', fc)
# what follows finish-intent phrases in loop samples (pre-onset): next 12 chars
nxt = {}
for r in dl.itertuples():
    s_ = samples[r.i]; t = s_['text']
    for m in fin_pat.finditer(t[:len(tok.decode(toks(t)[:r.onset]))]):
        e = t.find('\n', m.end())
        if e < 0: continue
        follow = t[e:e+12]
        key = follow.replace(' ', '_')
        nxt[key] = nxt.get(key, 0) + 1
print(sorted(nxt.items(), key=lambda x:-x[1])[:15])
```