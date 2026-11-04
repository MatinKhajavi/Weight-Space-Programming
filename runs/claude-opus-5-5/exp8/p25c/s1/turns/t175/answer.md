E8 failed badly. Accuracy on D fell to 14.1% (E6: 43.2%) and the cut share rose to 85% (E6: 48%), close to the unedited model (10.4%, 90%). My hypothesis was wrong. Under E6, the general output of L23 heads 5 and 6 does not sustain the leftover loops; it is needed for E6's loop-breaking effect. That is the opposite of what ablating the same heads in the original model suggested (cut 90% → 88%). I won't keep E8, but this strong interaction is informative about how E6 works. I'll compare gen_24 with gen_16 (E6, same seeds): paired accuracy, loop openers in E8's cut-offs (has the "Wait"/"So" lock returned, or is this a new kind of loop?), and loop periods. Then I'll restore E6 so a later pass can't run the wrong weights by accident.

```python
# Diagnose E8 failure: paired vs E6 (gen_16), loop openers/periods in E8 cut-offs; then restore E6 as current edit
g24=S('gen_24.jsonl'); g16=S('gen_16.jsonl')
d=[(b['correct']-a['correct']) for i in g16 for a,b in zip(g16[i][1],g24[i][1])]
print('E8-E6 paired', round(100*np.mean(d),1), '+-', round(100*np.std(d)/np.sqrt(len(d)),1), 'w->r', sum(x>0 for x in d), 'r->w', sum(x<0 for x in d))
fin=[s for src,ss in g24.values() for s in ss if not s['cut']]; print('E8 finished', len(fin), 'fin-acc', round(np.mean([s['correct'] for s in fin]),3))
def loop_first_words(G):
    c=Counter(); pers=[]; n=0
    for src,ss in G.values():
        for s in ss:
            if not s['cut']: continue
            p,o=find_loop(s['text'])
            if p is None: continue
            n+=1; pers.append(p); seg=s['text'][o:o+p]
            for ln in seg.split('\n'):
                w=ln.strip().lstrip('*').strip().split(' ')[0] if ln.strip() else ''
                if w: c[w[:12]]+=1
    return n, np.percentile(pers,[25,50,75]).round(0), c.most_common(14)
for nm,G in [('E6',g16),('E8',g24)]:
    n,pq,cm=loop_first_words(G); print(nm,'exact loops',n,'period q',pq); print('  ', [(asc(w),k) for w,k in cm])
for L in (15,19,23): save_edit(pfx%L, torch.load(f'/work/best_E6/L{L}_o_proj.pt'))
print('restored E6:', float((load(pfx%23)-WE6_23).abs().max()))
```