The loop table gives character offsets: `loopstart` is the start of the exact repetition and `period` its length in characters. Typical loops are about 45 tokens long and start around token 6,700 (range 970 to 32,600). The first example is instructive: the trivial problem "Compute sin 0°" ends up in an exact loop after 8,164 tokens, repeating fragments like "I will use `?_`." This suggests the long pre-loop phase matters too, and that the model may have been circling well before the exact repetition. Before building the measurement set, I want to see what the thinking looks like on the way into the loop for this text. If it is near-repetitive deliberation, "loop onset" is gradual and the copy heads are probably engaged long before exact repetition. In that case the edit should weaken self-copying generally, not only inside exact loops.

```python
# Hypothesis check: is the pre-loop phase already near-repetitive (gradual onset)? Read short excerpts of one looping text
t0 = c2.iloc[0]['text']; ls = int(c2.iloc[0]['loopstart'])
for pos in [600, 3000, 8000, 15000, 22000, ls-600, ls-250]:
    print(pos, '|', repr(t0[pos:pos+240]))
# how much of the pre-loop text is made of lines that already occurred earlier (line-level self-repetition)?
def line_rep_profile(text, upto, nb=8):
    lines = text[:upto].split('\n'); seen=set(); flags=[]
    for ln_ in lines:
        s_ = ln_.strip()
        flags.append(1 if (len(s_)>15 and s_ in seen) else 0); 
        if len(s_)>15: seen.add(s_)
    fl = np.array(flags); ch = np.array_split(fl, nb)
    return [round(c.mean(),2) for c in ch]
print('line-repeat share by eighth of pre-loop text:', line_rep_profile(t0, ls))
```