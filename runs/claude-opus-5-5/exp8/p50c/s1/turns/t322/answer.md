The C edit is restored and verified, and its three tensors match /work/best_C_x2 exactly. The balance forensics are informative. Across 3,584 neurons per layer, down-column and up-row norms are balanced to within 2–7% (median absolute deviation, log scale), as expected from weight decay. The brake neuron 22.573 sits at z = −1.1, entirely ordinary, and the copy heads' o/v norm ratios are ordinary too. So nothing suggests a halved brake or amplified heads: the looping is a natural property of this model, and the "×2 optimal" dose is not evidence of tampering.

The paired F23 analysis came out empty because each record is a problem holding a list of `samples`. The DeltaNet scan crashed with an out-of-memory error during memory-mapping, almost certainly because old kernel objects fill memory.

Now I will:
- free the large leftover variables;
- redo the paired analysis at sample level for F23 and all earlier C-variants, as transition tables against C;
- build X8 = C + head 19.5 zeroed. It is the only one of the four remaining loop-copy heads under C not yet tested (loop-copy mass .46; 11.1, 23.4 and 23.7 were tested).

If X8 behaves like 23.4 (cuts turned into wrong answers, no net gain), the head-ablation route is exhausted.

```python
# R2 hypothesis: (a) old kernel objects exhaust memory -> free; (b) records hold per-problem 'samples' -> proper paired F23 vs C + transition tables of all C-variants; (c) build X8 = C + head 19.5 zeroed.
import gc, os, json, re, numpy as np, torch, collections
P='model.language_model.layers.'
def sizeof(v, depth=0):
    if isinstance(v,np.ndarray): return v.nbytes
    if torch.is_tensor(v): return v.element_size()*v.nelement()
    if depth>2: return 0
    if isinstance(v,dict): return sum(sizeof(x,depth+1) for x in list(v.values())[:5000])
    if isinstance(v,(list,tuple)): return sum(sizeof(x,depth+1) for x in list(v)[:5000])
    if isinstance(v,str): return len(v)
    return 0
keep={'C_TARGET','BAL','NAMES','tok','P','In','Out'}
big=sorted([(sizeof(v),k) for k,v in list(globals().items()) if not k.startswith('_') and not callable(v) and k not in keep and not isinstance(v,type(os))], reverse=True)
print('top vars GB:', [(k, round(s/1e9,2)) for s,k in big[:10]])
freed=0
for s,k in big:
    if s>3e7: del globals()[k]; freed+=s
for k in list(globals()):
    if re.fullmatch(r'_\d+',k): del globals()[k]
if isinstance(globals().get('Out'),dict): globals()['Out'].clear()
gc.collect()
print('freed GB', round(freed/1e9,2), [l for l in open('/proc/self/status').read().split('\n') if l.startswith(('VmRSS','VmSize'))])
R=lambda p:[json.loads(l) for l in open(p)]
s0=R('/work/gen_38.jsonl')[0]['samples'][0]
print('sample keys:', {k:(v if not isinstance(v,(str,list,dict)) else type(v).__name__+str(len(v))) for k,v in s0.items()})
def st(s):
    c=s.get('correct', s.get('grade', s.get('is_correct')))
    if isinstance(c,str): c=c.strip().lower() in ('1','true','correct','yes')
    if c is True or (isinstance(c,(int,float)) and not isinstance(c,bool) and c>0.5): return 'ok'
    for k in ('cut','truncated','cut_at_max_tokens','hit_max'):
        if s.get(k): return 'cut'
    f=str(s.get('failure', s.get('finish_reason', s.get('status',''))))
    if 'cut' in f or 'length' in f or 'max' in f: return 'cut'
    return 'wrong'
def flat(*paths):
    out={}
    for p in paths:
        for r in R(p):
            for j,s in enumerate(r['samples']): out[(r['id'],j)]=st(s)
    return out
CS=flat('/work/gen_13.jsonl','/work/gen_16_edited.jsonl')
print('C n', len(CS), 'acc', round(float(np.mean([v=='ok' for v in CS.values()])),3), dict(collections.Counter(CS.values())))
var={'X1(23.4)':'gen_22','X3(11.1)':'gen_25','X4(23.7)':'gen_26','X6(maybe-brk)':'gen_29','X7(23.6x1.5)':'gen_34','Va(brk2.5)':'gen_35','Vb(heads.5)':'gen_36','F15':'gen_37','F23':'gen_38'}
rows=[]; RES={}
for name,g in var.items():
    V=flat(f'/work/{g}.jsonl'); ks=[k for k in V if k in CS]
    d=np.array([(V[k]=='ok')-(CS[k]=='ok') for k in ks],float); t=collections.Counter((CS[k],V[k]) for k in ks)
    RES[name]=(d.mean()*100, d.std(ddof=1)/np.sqrt(len(d))*100)
    rows.append(f"{name}: n{len(ks)} acc {np.mean([V[k]=='ok' for k in ks])*100:.1f} diff {d.mean()*100:+.1f} SE {RES[name][1]:.1f} | ok>w {t[('ok','wrong')]} ok>c {t[('ok','cut')]} w>ok {t[('wrong','ok')]} c>ok {t[('cut','ok')]} w>c {t[('wrong','cut')]} c>w {t[('cut','wrong')]}")
print('\n'.join(rows))
try:
    an=[n for n in NAMES if n.endswith('A_log')]
    allA=np.concatenate([load_original(n).numpy().ravel() for n in an]); mA=np.median(allA); sA=1.4826*np.median(np.abs(allA-mA))
    dl=[]
    for n in an:
        L=n.split('layers.')[1].split('.')[0]; a=load_original(n).numpy().ravel(); n2=n.replace('A_log','dt_bias')
        dt=load_original(n2).numpy().ravel() if n2 in NAMES else np.zeros(1)
        dl.append(f'L{L} A {a.min():.1f}/{np.median(a):.1f}/{a.max():.1f} dt {dt.min():.1f}/{dt.max():.1f} fl{[i for i in range(len(a)) if abs(a[i]-mA)/sA>3]}')
    print(' | '.join(dl))
except Exception as e: print('deltanet scan err', e)
o19=load_original(P+'19.self_attn.o_proj.weight').clone(); o19[:,5*256:6*256]=0
for k,v in C_TARGET.items(): save_edit(k,v)
save_edit(P+'19.self_attn.o_proj.weight', o19)
print('edited:', [n.replace(P,'L') for n in edited()])
f23=RES['F23']
with open('/work/notes.md','a') as fh:
    fh.write(f"\n## Session 6\n- PASS 32 = gen_38: F23 (brake x2 + only 23.0/23.5 zeroed) SH512: acc 58.4, paired vs C {f23[0]:+.1f} SE {f23[1]:.1f}, cut 35.7 => both routes needed; C best. C restored+verified (matches best_C_x2).\n- Balance forensics: log(down-col/up-row norm) MAD 0.02-0.07 per layer (tight weight-decay balance); 22.573 z=-1.1 (ordinary), copy-head o/v ordinary (L23 range .73-1.13) => NO rescaling signature; loops are natural behaviour.\n- X8 = C + 19.5 zeroed built; pass 33 planned (edited, SH512).\n")
sh=json.load(open('/work/SH512_ids.json')); dump=json.dumps(sh)
print('SH512 type', type(sh).__name__, 'len', len(sh), 'dump chars', len(dump))
print(dump if len(dump)<5200 else dump[:300])
```