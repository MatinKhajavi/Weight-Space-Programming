The overshoot is worse. Scaling A's three heads to -0.5 gives +13.7 points (SE 2.7) against +22.7 at 0, wrong answers triple (28 versus 9), and cut-offs don't fall further (64.5% versus 61.3%). The dose-response so far is: scale 1 → 0, scale 0 → +22.7, scale -0.5 → +13.7. So these heads' normal copying is useful once inverted, and zero is near the best point for them. More importantly, the remaining ~62% cut-off floor hasn't moved under any variant (A, B, A at -0.5), so whatever sustains the late loops lies outside these heads. To localise it I'll run a teacher-forced pass with edit A on two windows of edit A's own looping texts (about 1,100 tokens each, half pre-loop and half loop), collecting all hidden states, MLP activations and attention. A per-layer DLA of the in-loop excess will show whether the remaining copying is in other attention heads or in the DeltaNet layers. This block finds suitable ASCII-only windows and prints them for the request.

```python
# Hypothesis: (setup) pick 2 ASCII-only windows from edit-A loop samples (pre-loop ~1800 chars + loop ~2200 chars)
with open('/work/notes.md','a') as f: f.write("""
- PASS 7 = gen_10 (both, S1): A(-0.5) 26.2% (gsm 28.9, math 23.4) cut 64.5% wrong 28; diff +13.7 SE 2.7. Dose: scale1 +0, scale0 +22.7, scale-0.5 +13.7 => 0 near optimum; anti-copy hurts number copying. Cut floor ~62% unchanged by A/B/A-0.5.
""")
wins=[]
for _,x in g6e[g6e.cut&g6e.on.notna()].iterrows():
    if not (40<=x.P<=200) or x.on<4000: continue
    t=RA[x.id]['samples'][0]['text']; w=t[int(x.on)-1800:int(x.on)+2200]
    if all(ord(c)<128 for c in w) and w.count('    ')<120:
        wins.append((x.id,int(x.on),int(x.P),w))
print(len(wins),[(a,b,c,len(tok.encode(d).ids)) for a,b,c,d in wins[:6]])
```