X6 (C plus the doubled Maybe brake) scored 58.8% on SH512, against 60.7% for C on the same problems and seeds. Cut answers barely moved (24.2% vs about 24.8%) and wrong answers rose (85 vs 75). So this looks like no gain, possibly a small loss. Before concluding I need the paired difference with its SE and the outcome cross-tab. I also need the doubt-word rates, to check that the brake actually collapsed "Maybe" lines and to see where the doubt went. If Maybe collapsed and accuracy did not improve, the Maybe spirals are a symptom: the doubt is re-expressed in other words, unlike Wait, whose lines also fed copy-loops. I will revert to C in the same run, since there is no evidence of a gain.

```python
# Evaluate X6 (Maybe brake x2) vs C paired on SH512; check maybe-family collapse and where doubt went; revert to C
X6=smap(['/work/gen_29.jsonl']); keys=[k for k in X6 if k in C]
d=np.array([(X6[k][0]=='ok')-(C[k][0]=='ok') for k in keys],dtype=float)
ct=collections.Counter((C[k][0],X6[k][0]) for k in keys)
print('n',len(keys),'C acc %.3f X6 acc %.3f diff %+.3f SE %.3f'%(np.mean([C[k][0]=='ok' for k in keys]),np.mean([X6[k][0]=='ok' for k in keys]),d.mean(),d.std(ddof=1)/np.sqrt(len(d))))
print({f'{a}->{b}':ct[(a,b)] for a in ('ok','cut','wrong') for b in ('ok','cut','wrong')})
for src in ('gsm8k','math'):
    kk=[k for k in keys if C[k][1]==src]; dd=np.array([(X6[k][0]=='ok')-(C[k][0]=='ok') for k in kk],float)
    print(src, 'diff %+.3f SE %.3f'%(dd.mean(), dd.std(ddof=1)/np.sqrt(len(dd))))
def rates2(paths):
    tot=0; c=collections.Counter()
    for p in paths:
        for r in lg(p):
            for s in r['samples']:
                tx=s['text']; tot+=s['tokens']
                for nm,pat in [('LI_Maybe',r'\n[ \t]*(?:[\*\-]\s+)?(?:\d+\.\s+)?Maybe\b'),('maybe',r'\bmaybe\b'),('Perhaps',r'\b[Pp]erhaps\b'),('LI_If',r'\n[ \t]*(?:[\*\-]\s+)?If\b'),('LI_Or',r'\n[ \t]*(?:[\*\-]\s+)?Or\b'),('Altern',r'\bAlternatively\b'),('Could',r'\bCould\b'),('Is it',r'\bIs it\b'),('Possib',r'\b[Pp]ossib')]:
                    c[nm]+=len(re.findall(pat,tx))
    return {k:round(1000*v/tot,2) for k,v in c.items()}
print('C  ', rates2(['/work/gen_13.jsonl','/work/gen_16_edited.jsonl']))
print('X6 ', rates2(['/work/gen_29.jsonl']))
W22=load('model.language_model.layers.22.mlp.down_proj.weight').clone(); W22[:,387]=W22o[:,387]; save_edit('model.language_model.layers.22.mlp.down_proj.weight',W22)
print('reverted to C:', bool(torch.allclose(load('model.language_model.layers.22.mlp.down_proj.weight')[:,387],W22o[:,387])), bool(torch.allclose(load('model.language_model.layers.22.mlp.down_proj.weight')[:,573],2*W22o[:,573])))
```