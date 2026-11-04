Character-to-token ratios vary a lot between candidates, from about 1.6 to 6 characters per token. Each window must stay under 2,048 tokens and the two together under my roughly 10k-character print limit. I'll take two math-heavy windows (intermediate_algebra_948 and precalculus_742), trim them to about 3,000-3,200 characters, check token counts, and print them as JSON strings to paste into the request. The region boundaries (loop onset inside each window) are recorded so I can analyse the returned arrays.

```python
# Hypothesis: (setup) trim two windows to <=2000 tokens and print as JSON strings for the TF request
sel_w=[]
for wid,(b,a) in [('math_train_intermediate_algebra_948',(1400,1600)),('math_train_precalculus_742',(1500,1700))]:
    x=g6e[g6e.id==wid].iloc[0]; t=RA[wid]['samples'][0]['text']; on=int(x.on)
    w=t[on-b:on+a]; sel_w.append(dict(id=wid,on_in_win=b,P=int(x.P),text=w,ntok=len(tok.encode(w).ids)))
print([(s['id'],s['ntok'],s['P']) for s in sel_w])
print(json.dumps([s['text'] for s in sel_w]))
```