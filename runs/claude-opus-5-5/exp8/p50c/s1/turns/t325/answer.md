X8 failed badly. Zeroing head 19.5 on top of C dropped accuracy from 60.7% to 41.6% on the same 512 items (wrong answers 75 → 124, cut share 24.6% → 33.4%). So 19.5 is essential for general reasoning, not a loop driver.

This completes the head map. Only the three chosen copy heads (15.7, 23.0, 23.5) are beneficial to remove; every other loop-participating head is neutral or needed:
- 23.4: 0.0
- 23.7: −3.5
- 11.1: −11.3
- 19.5: −19.1

I now restore C. Before deciding how to spend the last 17 passes, I need to know what C's remaining loops look like. I will characterise every cut answer among C's 1,280 answers (SH512 plus held-out H2–H4) on these measures:
- whether the tail is exactly periodic;
- the length of the repeating unit and its first word;
- whether the unit contains digits, "Wait", or a boxed answer;
- whether the reference answer appears inside the loop or before its onset.

If a sizable share of loops repeat a unit that already contains the right answer, a loop-specific exit is the one remaining failure worth targeting. If the units are mostly unsolved computations in hard problems, the remaining cuts reflect capability and C is final.

```python
# R4 hypothesis: X8 shows 19.5 is essential -> restore C; characterize remaining C loops (periodicity, unit length/first word, digits, answer-in-unit) over all 1280 C answers to see if a specific loop type is left to target.
import json, re, collections, numpy as np
for k,v in C_TARGET.items(): save_edit(k,v)
for n in list(edited()):
    if n not in C_TARGET: drop_edit(n)
okC=all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
print('edited:', [n.replace(P,'L') for n in edited()], 'C verified', okC)
V=flat('/work/gen_39.jsonl'); ks=[k for k in V if k in CS]
dx=np.array([(V[k]=='ok')-(CS[k]=='ok') for k in ks],float); t=collections.Counter((CS[k],V[k]) for k in ks)
x8=f"X8(19.5): n{len(ks)} acc {np.mean([V[k]=='ok' for k in ks])*100:.1f} diff {dx.mean()*100:+.1f} SE {dx.std(ddof=1)/np.sqrt(len(dx))*100:.1f} | ok>w {t[('ok','wrong')]} ok>c {t[('ok','cut')]} w>ok {t[('wrong','ok')]} c>ok {t[('cut','ok')]} w>c {t[('wrong','cut')]} c>w {t[('cut','wrong')]}"
print(x8)
def tail_loop(tx, maxp=2500, minrep=3):
    n=len(tx)
    for p in range(8, min(maxp, n//minrep)):
        seg=tx[n-minrep*p:]
        if seg[:-p]==seg[p:]:
            i=n-minrep*p
            while i-p>=0 and tx[i-p:i]==tx[i:i+p]: i-=p
            while i>0 and tx[i-1]==tx[i-1+p]: i-=1
            return p,i
    return None
rows=[]
for f in ['gen_13','gen_16_edited','gen_27_edited','gen_32_edited','gen_33_edited']:
    for r in R(f'/work/{f}.jsonl'):
        for s in r['samples']: rows.append((r['id'], r.get('source',''), str(r.get('answer','')).strip(), s))
cuts=[x for x in rows if x[3].get('cut')]
print('C answers', len(rows), 'cut', len(cuts), 'by src', dict(collections.Counter(x[1] for x in cuts)))
res=[]
for pid,src,ans,s in cuts:
    tx=s['text']; Lp=tail_loop(tx)
    if Lp is None: res.append(dict(src=src, per=False)); continue
    p,i=Lp; unit=tx[i:i+p]; j=unit.find('\n'); u=(unit[j+1:]+unit[:j+1]) if j>=0 else unit
    m=re.match(r'\s*([A-Za-z]+|\S)', u); fw=m.group(1) if m else '?'
    longans=len(ans)>=2
    res.append(dict(src=src, per=True, p=p, onset=i/len(tx), fw=fw, dig=bool(re.search(r'\d',u)), wait=('Wait' in u or 'wait' in u),
                    ans_in=(longans and ans in u), longans=longans, boxed=('boxed' in u), ans_before=(longans and ans in tx[:i]), nl=u.count('\n')))
summ_lines=[]
for src in ['gsm8k','math']:
    Rs=[x for x in res if x['src']==src]; Pq=[x for x in Rs if x['per']]; La=[x for x in Pq if x['longans']]
    if not Pq: continue
    line=(f"{src}: cut {len(Rs)} periodic {len(Pq)} | unit chars med {np.median([x['p'] for x in Pq]):.0f} (p10 {np.percentile([x['p'] for x in Pq],10):.0f}, p90 {np.percentile([x['p'] for x in Pq],90):.0f}) lines/unit med {np.median([x['nl'] for x in Pq]):.0f} | onset frac med {np.median([x['onset'] for x in Pq]):.2f} (p10 {np.percentile([x['onset'] for x in Pq],10):.2f}) | digits {np.mean([x['dig'] for x in Pq]):.2f} Wait-in-unit {np.mean([x['wait'] for x in Pq]):.2f} boxed-in-unit {np.mean([x['boxed'] for x in Pq]):.2f} | ref(len>=2, n={len(La)}) in unit {np.mean([x['ans_in'] for x in La]) if La else 0:.2f} before onset {np.mean([x['ans_before'] for x in La]) if La else 0:.2f}")
    fws=collections.Counter(x['fw'] for x in Pq).most_common(12)
    print(line); print('   unit first words:', fws); summ_lines.append(line+' | fw '+str(fws[:6]))
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASS 33 = gen_39: {x8} => 19.5 ESSENTIAL (worst ablation). Head map complete: only 15.7/23.0/23.5 beneficial; C restored (verified {okC}).\n- Remaining C loops (1280 C answers, cut texts): " + ' || '.join(summ_lines) + "\n")
```