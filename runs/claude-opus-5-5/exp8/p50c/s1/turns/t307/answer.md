The fresh MATH set H4 also confirms the repair. The original gets 7.8% (93.75% of answers cut) and C gets 55.1%. The paired difference is +47.3 points (SE 3.2), with 123 wrong→right and only 2 right→wrong. Across the four held-out sets (H1, H2, H3, H4; 1,024 paired answers) the gain is consistently +46 to +49 points. The H4 breakdown also shows what limits MATH now: only 12.5% of answers are wrong while 32.4% are still cut at the token limit. Earlier, pooled over both benchmarks, the reference answer appeared before the loop in more cut texts than chance (56% vs 31%). If many MATH cut-offs happen after the model has already reached, or even boxed, the correct answer, failure to commit is a sizeable, specific target. I will measure it on MATH cut texts with distinctive reference answers (fractions, roots, multi-digit numbers), so chance matches stay rare. The control is the rate at which other problems' answers appear in the same texts. I will log H3/H4 in the notes in the same run.

```python
# How often do C's MATH cut-offs contain the correct answer (or a boxed correct answer) before the loop? control: other problems' answers
with open('/work/notes.md','a') as f:
    f.write("- PASS 26 = gen_32 (both, H3 = 256 fresh GSM8K): orig 16.8 cut 87.1 | C 62.5 cut 14.5 wrong 22.7; diff +45.7 SE 3.2 (W->R 118, R->W 1).\n- PASS 27 = gen_33 (both, H4 = 256 fresh MATH, MATH-500 level mix): orig 7.8 cut 93.8 | C 55.1 cut 32.4 wrong 12.5; diff +47.3 SE 3.2 (W->R 123, R->W 2). Held-out total H1-H4: +45.7..+49.2 every set.\n- Weight forensics: copy heads and 22.573 have ordinary norms (no tampering signature); only low-norm unused neurons are outliers.\n")
def nrm(a): 
    a=str(a).replace('\\dfrac','\\frac').replace('\\tfrac','\\frac').replace('\\left','').replace('\\right','').replace(' ','').replace('$','')
    return a
def distinctive(a): return ('\\frac' in a or '\\sqrt' in a or '\\pi' in a or (re.fullmatch(r'-?\d+',a) is not None and len(a.lstrip('-'))>=2))
rows=[]
for p in ['/work/gen_33_edited.jsonl','/work/gen_27_edited.jsonl','/work/gen_13.jsonl','/work/gen_16_edited.jsonl']:
    R=lg(p); refs=[nrm(r['answer']) for r in R if r['source']=='math']
    for r in R:
        if r['source']!='math': continue
        ref=nrm(r['answer'])
        if not distinctive(ref): continue
        for s in r['samples']:
            if status(s)!='cut': continue
            tx=s['text'].replace(' ','').replace('\\dfrac','\\frac').replace('\\left','').replace('\\right','')
            cl=char_loop(s['text']); cut_at=len(tx) if cl is None else int(len(tx)*cl[1]/len(s['text']))
            pre=tx[:cut_at]
            others=[o for o in random.Random(len(rows)).sample(refs,12) if o!=ref and distinctive(o)][:5]
            rows.append((ref in pre, ('\\boxed{'+ref+'}') in pre, np.mean([o in pre for o in others]) if others else np.nan, cl is not None))
rows=np.array(rows,dtype=float)
print('MATH cut texts with distinctive ref:', len(rows), '| exact-loop share %.2f'%rows[:,3].mean())
print('ref appears before onset %.2f | boxed ref before onset %.2f | control (other refs) %.2f'%(rows[:,0].mean(), rows[:,1].mean(), np.nanmean(rows[:,2])))
```