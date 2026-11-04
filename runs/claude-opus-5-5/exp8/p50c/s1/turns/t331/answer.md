Turning the L23 copy heads into anti-induction heads backfired. X10 scored 51.0% against C's 60.7% on the same 512 items (−9.7), and the cut share rose from 24.6% to 35.0% instead of falling. Reversed heads destabilise generation rather than act as a repetition penalty.

This completes a clean dose-response for heads 23.0/23.5, with 15.7 zeroed and the brake at ×2. Accuracy peaks sharply at exactly 0, as does the cut share:

| Head scale | ×1 | ×0 | ×−0.5 |
|---|---|---|---|
| Accuracy | 54.9% | 60.7% | 51.0% |
| Cut share | 37.5% | 24.8% | 35.0% |

Head manipulations, brake doses, the Maybe-brake and anti-copying are now all exhausted; C is the optimum along every axis tested.

Next I restore C and log X10. I also want to recall which teacher-forced analyses already exist (for example whether C's effect on the likelihood of reference solutions was measured). That decides whether the remaining passes go to a capability-preservation check, the fresh confirmations H6/H7, or a final search at the post-answer decision point behind GSM exit failures.

```python
# R7 hypothesis: X10 (anti-induction) hurts -> restore C; recall which TF analyses exist (reference-solution likelihood etc.) to plan remaining passes.
for k,v in C_TARGET.items(): save_edit(k,v)
for n in list(edited()):
    if n not in C_TARGET: drop_edit(n)
okC=all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
V=flat('/work/gen_42.jsonl'); ks=[k for k in V if k in CS]
dx=np.array([(V[k]=='ok')-(CS[k]=='ok') for k in ks],float); t=collections.Counter((CS[k],V[k]) for k in ks)
x10=f"X10(23.0/23.5 x-0.5): acc {np.mean([V[k]=='ok' for k in ks])*100:.1f} diff {dx.mean()*100:+.1f} SE {dx.std(ddof=1)/np.sqrt(len(dx))*100:.1f} | ok>w {t[('ok','wrong')]} ok>c {t[('ok','cut')]} w>ok {t[('wrong','ok')]} c>ok {t[('cut','ok')]} w>c {t[('wrong','cut')]} c>w {t[('cut','wrong')]}"
print('C restored', okC, [n.replace(P,'L') for n in edited()]); print(x10)
nt=open('/work/notes.md').read()
for l in nt.split('\n'):
    if any(s in l for s in ('fwd_11','fwd_19','corpus','TF ','teacher','likelihood','logprob','ref sol','reference sol')):
        print('>>', l.encode('ascii','replace').decode()[:260])
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASS 35 = gen_42: {x10}. L23-head dose under brake (15.7=0): x1 54.9 / x0 60.7 / x-0.5 51.0 (cut 37.5/24.8/35.0) => sharp optimum at 0; anti-induction destabilises. C restored (verified {okC}).\n")
```