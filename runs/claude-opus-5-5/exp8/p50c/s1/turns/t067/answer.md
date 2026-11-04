Under edit A, GSM8K still has 52% cut-offs even though these are grade-school problems. Level 1 MATH is at 27%, so the remaining loops are not mainly a difficulty effect; the pathology persists on easy problems. Edit B lost accuracy at every MATH level from 2 to 4 and on GSM8K, confirming those extra heads carry useful copying. To see what still drives looping under A, I'll read the onset context and loop unit of several edit-A GSM8K cut samples. In particular, has the model already computed the answer and become stuck verifying it, or is it stuck mid-computation?

```python
# Hypothesis: edit-A GSM8K loops happen after the answer is found (stuck verifying/finishing), not mid-computation
RA={json.loads(l)['id']:json.loads(l) for l in open('/work/gen_6_edited.jsonl')}
gc=g6e[(g6e.src=='gsm8k')&g6e.cut&g6e.on.notna()]
for _,x in gc.head(6).iterrows():
    t=RA[x.id]['samples'][0]['text']; on,P=int(x.on),int(x.P); ans=PRB[x.id]['answer']
    print('==',x.id,'ans',ans,'onTok~',len(tok.encode(t[:on]).ids),'P',P,'ans in pre-text:',(' '+str(ans)) in t[:on])
    print('  pre:',repr(t[on-200:on]).encode('ascii','replace').decode()[:230]); print('  unit:',repr(t[on:on+min(P,160)]).encode('ascii','replace').decode())
```