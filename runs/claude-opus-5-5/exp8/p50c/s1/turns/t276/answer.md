The "Final" neurons are lexical output neurons for the instruction text. They fire on the prompt's "put your **final** answer within \boxed{}" (17.838 at +2.3 before " final", 19.734 at +2.8 before " answer") and are essentially silent inside the thinking loops (5th–95th percentiles within ±0.15). So inside loops there is no latent conclusion impulse for amplification to tip over; the commit failure is not a near miss. Amplifying them would only touch contexts where the model already intends to write "Final". I drop this route.

Before settling on the endgame, one question remains about the current brake's dose. At ×2 it has nearly eliminated line-initial "Wait" (0.2%). Going to ×3 only matters if its other doubt targets, "Actually" (z 3.9) and "Hmm" (z 3.6), are frequent loop-starters under C. Earlier, "Maybe" lines were 3× more frequent in failing texts but suppressing them did not help. I will compare line-initial rates per 1,000 tokens of Actually, Hmm, Maybe, Alternatively, But and Let between C's cut and correct texts. If Actually/Hmm are both common and enriched in failures, a ×3 dose is worth a pass; otherwise I skip it.

```python
# Hypothesis check before a x3 brake test: are 'Actually'/'Hmm' line starts frequent and enriched in C's failing (cut) texts?
def rate(texts, w):
    return np.mean([1000*len(re.findall(r'(?m)^\s*(?:[*-]\s*)?(?:\*\*)?'+w+r'\b', t.split('</think>')[0]))/max(1,len(t)/4) for t in texts])
cutT = [t for _,_,t in cutC]; okT = [r['samples'][0]['text'] for r in recC if r['samples'][0]['correct']]
wrT = [r['samples'][0]['text'] for r in recC if (not r['samples'][0]['correct']) and not r['samples'][0]['cut']]
for w in ['Actually','Hmm','Maybe','Alternatively','But','Let','Wait','So','This']:
    print(f'{w:14s} per1k  cut {rate(cutT,w):.2f}  wrong {rate(wrT,w):.2f}  ok {rate(okT,w):.2f}')
```