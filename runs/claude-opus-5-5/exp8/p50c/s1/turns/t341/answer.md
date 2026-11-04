BR (C with the brake's non-Wait remainder doubled and its Wait component left at the original level) scored 56.1% on SH512, against 60.7% for C, with cuts at 35.9% instead of 24.6%. That is well below C but far above what the heads-only edit A suggests (35.2% on S1, where C got 60.2%). So the picture is asymmetric:
- The Wait component alone fully reproduces C (BW).
- The remainder alone recovers much of the effect but not all of it.
- Together they saturate: BW plus the remainder equals C.

Two explanations fit. Either the remainder lowers "Wait" indirectly (through layer 23 or through Wait-like tokens outside the two exact Wait directions), or it suppresses other doubt-starting tokens ("Hmm", "But", "Actually", "Alternatively"). I can separate these without a pass:
- measure line-initial "Wait" and other starter rates on the same S1 problems for the original model, A, BR, BW and C;
- compute the direct logit effect of the Wait component and of the remainder on a list of starter tokens.

If BR's line-initial Wait rate falls well below A's, the remainder works by indirectly suppressing Wait. If BR keeps A-level Wait but suppresses other starters, the neuron is a broader "doubt-family brake". This run also computes BR's paired difference with C.

```python
# R12 hypothesis: the brake column's non-Wait remainder helps by (i) indirectly lowering line-initial 'Wait' or (ii) suppressing other doubt starters; distinguish via line-initial starter rates on S1 (orig/A/BR/BW/C) and direct logit effects of colW vs remainder on starter tokens.
import re, json, collections, numpy as np, torch
V=flat('/work/gen_47.jsonl'); ks=[k for k in V if k in CS]
dx=np.array([(V[k]=='ok')-(CS[k]=='ok') for k in ks],float); t=collections.Counter((CS[k],V[k]) for k in ks)
br=f"BR(remainder-only brake): acc {np.mean([V[k]=='ok' for k in ks])*100:.1f} diff {dx.mean()*100:+.1f} SE {dx.std(ddof=1)/np.sqrt(len(dx))*100:.1f} | ok>w {t[('ok','wrong')]} ok>c {t[('ok','cut')]} w>ok {t[('wrong','ok')]} c>ok {t[('cut','ok')]} w>c {t[('wrong','cut')]} c>w {t[('cut','wrong')]}"
print(br)
S1=set(json.load(open('/work/S1_ids.json'))) if not isinstance(json.load(open('/work/S1_ids.json')),dict) else set()
W=['Wait','Maybe','Hmm','Actually','But','Alternatively','Let','So','However','Hold','Therefore']
def li_rates(path, ids=None):
    cnt=collections.Counter(); nl=0; acc=[]; cut=[]
    for r in R(path):
        if ids is not None and r['id'] not in ids: continue
        for s in r['samples']:
            acc.append(bool(s.get('correct'))); cut.append(bool(s.get('cut')))
            for l in s['text'].split('\n'):
                l2=re.sub(r'^[\s*\-#>]*(\d+\.\s*)?(\*\*)?\s*','',l)
                if not l2: continue
                nl+=1; m=re.match(r'([A-Za-z]+)',l2)
                if m and m.group(1) in W: cnt[m.group(1)]+=1
    return np.mean(acc), np.mean(cut), nl, {w:round(cnt[w]/nl*1000,1) for w in W}
for nm,pth in [('orig','/work/gen_6_original.jsonl'),('A(heads)','/work/gen_6_edited.jsonl'),('BR','/work/gen_47.jsonl'),('BW','/work/gen_46.jsonl'),('C','/work/gen_13.jsonl')]:
    a,c,nl,rt=li_rates(pth, S1 if nm in ('BR','BW') else None)
    print(f"{nm:9s} S1 acc {a*100:.1f} cut {c*100:.1f} lines {nl} | per-1k-lines:", rt)
rem=col573-colW
WU=load_original('model.language_model.embed_tokens.weight')
rows_=[]
for w in ['Wait',' Wait','wait',' wait','Hmm',' Hmm','Actually',' Actually','But',' But','Alternatively','Maybe',' Maybe','Let',' Let','So',' So','However','Hold','Therefore','**','</think>']:
    ids=tok.encode(w).ids
    if len(ids)!=1: continue
    u=geff*WU[ids[0]]
    rows_.append(f"{w.strip() or w}{'(sp)' if w.startswith(' ') else ''}:{float(colW@u)*-5:+.2f}/{float(rem@u)*-5:+.2f}")
print('direct logit change at act=-5 (Wait-part / remainder):', ' '.join(rows_))
del WU
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASS 39 = gen_47: {br} (cut 35.9). => remainder alone recovers much but not all; Wait part alone = full C.\n")
```