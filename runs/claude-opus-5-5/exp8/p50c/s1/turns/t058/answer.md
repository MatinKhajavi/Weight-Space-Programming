The tool only accepts literal texts, so large teacher-forced corpora aren't practical: about 1.4 MB would have to go inline. I'll use teacher forcing only for small mechanistic checks (two short texts). Edits will be screened mainly with full-size generation passes, which measure the outcome that counts (accuracy, plus cut-off rate as a mechanism readout). The causal question is whether removing the loop-locking induction heads stops the verbatim loops. If so, does the model then finish and get answers right, or lose its ability to reason? Let me check the pools so I can plan screening on S and confirmation on H, and look at how the 44 correct answers close their thinking.

```python
# Hypothesis: none (planning) - pool sizes; how correct answers end their thinking (what precedes </think>)
pools=json.load(open('/work/pools.json')); print({k:(len(v) if hasattr(v,'__len__') else v) for k,v in pools.items()})
for r,s in ok[:6]:
    t=s['text']; i=t.find('</think>'); print('==',r['id'],int(s['tokens']),repr(t[max(0,i-160):i+12]).encode('ascii','replace').decode())
wc=[s['text'].count('Wait')/max(1,int(s['tokens']))*1000 for r,s in ok]; wl=[s['text'][:20000].count('Wait')/5 for r,s in cut_s]
print('Wait per 1k tok: correct',np.mean(wc).round(2),' cut(first~5k tok)',np.mean(wl).round(2))
```