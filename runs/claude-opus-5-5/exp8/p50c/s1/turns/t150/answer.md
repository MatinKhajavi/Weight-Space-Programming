Calibration shows the embedding-space copy score does not detect the late copy heads I already validated causally: 15.7, 23.0 and 23.5 all score about 0. It does find token copiers in layer 11 (11.1, 11.3, 11.4, 11.7) and copy-suppression heads (11.0, 11.6, 23.2). So this weight metric cannot localize the late-layer line recall, and the weak L21 scores prove nothing either way. Since inline teacher-forced texts are capped at a few kilobytes per pass, causal localization within L21 by teacher forcing would be slow. Before investing there, I want to size the prize better. Across both evaluations of the current edit (S1 and H1), I'll check how often failing samples had already produced the correct answer before going wrong, compared with correct samples and with a random-answer control. That separates "never solved it" from "solved it, then doubted or looped away".

```python
# Hypothesis: many C failures (wrong/cut) had already found the correct answer (over-doubting), unlike chance
allC=[]
for gname,G in [('g13',G13),('g16',E16)]:
    for r in G:
        for s in r['samples']:
            allC.append(dict(g=gname,id=r['id'],src=r['source'],ans=r['answer'],text=s['text'],ok=s['correct'] in (True,'True'),cut=s['cut'] in (True,'True'),tokens=int(s['tokens'])))
def first_num_pos(text, ans):
    a=str(ans).replace(',','').strip()
    try: v=float(a)
    except: return None
    for m in re.finditer(r'(?<![\d.])-?\d[\d,]*\.?\d*', text):
        try:
            if abs(float(m.group().replace(',','').rstrip('.'))-v)<1e-9: return m.start()
        except: pass
    return -1
rng=np.random.RandomState(1)
gs=[x for x in allC if x['src']=='gsm8k']
for cat,f in [('correct',lambda x:x['ok']),('wrong',lambda x:(not x['ok']) and not x['cut']),('cut',lambda x:x['cut'] and not x['ok'])]:
    xs=[x for x in gs if f(x)]
    fp=[first_num_pos(x['text'],x['ans']) for x in xs]
    has=[p is not None and p>=0 for p in fp]
    ctrl=[first_num_pos(x['text'], gs[rng.randint(len(gs))]['ans']) for x in xs]
    hc=[p is not None and p>=0 for p in ctrl]
    rel=[p/len(x['text']) for p,x in zip(fp,xs) if p is not None and p>=0]
    print(cat,'n',len(xs),'ref answer appears',round(np.mean(has),2),'control',round(np.mean(hc),2),'median rel pos',round(float(np.median(rel)) if rel else -1,2), 'median tokens',int(np.median([x['tokens'] for x in xs])))
```