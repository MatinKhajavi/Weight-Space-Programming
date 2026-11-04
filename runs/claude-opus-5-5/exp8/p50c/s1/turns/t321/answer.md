Pass 32 tested F23: the Wait-brake neuron 22.573 doubled, with only heads 23.0 and 23.5 zeroed. It scored 58.4% with 35.7% cut, against 60.7% and 24.8% cut for C on the same 512 items. Together with F15 (−5.9), this says both copy-head routes are needed and C, which removes all three heads, remains the best edit. The current weights are F23 because layer 15 is not edited, so the first job is to restore C exactly and check it against the saved copy in /work/best_C_x2. I will also compute F23's paired difference against C properly.

I have only 20 Python runs left, so this block also runs a cheap new check. If any component was rescaled after training, scale-symmetry "balance" ratios should flag it: down-column versus up-row norm per MLP neuron, and o-block versus v norm per attention head. It also matters that a dose of exactly ×2 was optimal for the brake neuron. If 22.573 shows up as a low-side outlier (z < −3), that supports a "halved component" hypothesis, and other outliers there become candidates for repair. If 22.573 looks ordinary and nothing functional stands out, I drop that idea.

```python
# R1 hypothesis: (a) restore C exactly; (b) F23 is below C in a paired comparison; (c) rescaled components would show as outliers of scale-symmetry balance ratios (down-col/up-row per neuron, o-block/v per head) -> check brake 22.573, copy heads, others.
import os, json, numpy as np, torch, collections
from safetensors.torch import load_file
P='model.language_model.layers.'
try:
    hs=[k for k,v in list(globals().items()) if callable(v) and not k.startswith('_') and getattr(v,'__module__','')=='__main__']
    print('helpers:', ', '.join(hs)[:900])
except Exception as e: print('helpers err', e)
nt=open('/work/notes.md').read(); print('notes len', len(nt)); print(nt[:900].encode('ascii','replace').decode())
def load_dir(dd):
    out={}
    for f in sorted(os.listdir(dd)):
        p=os.path.join(dd,f); base=f
        for ext in ('.safetensors','.pt','.pth','.npy'):
            if base.endswith(ext): base=base[:-len(ext)]
        try:
            if f.endswith('.safetensors'):
                for k,v in load_file(p).items(): out[k if k.startswith(('model.','lm_head')) else base]=v.float()
            elif f.endswith(('.pt','.pth')):
                obj=torch.load(p,map_location='cpu')
                if isinstance(obj,dict):
                    for k,v in obj.items(): out[k if str(k).startswith('model.') else base]=v.float()
                else: out[base]=obj.float()
            elif f.endswith('.npy'): out[base]=torch.from_numpy(np.load(p)).float()
        except Exception as e: print('load fail',f,e)
    return out
print('best_C_x2:', os.listdir('/work/best_C_x2'), '| edited dir:', os.listdir('/work/edited')[:6])
bestC=load_dir('/work/best_C_x2'); print('bestC keys:', [k.replace(P,'L') for k in bestC])
o15=load_original(P+'15.self_attn.o_proj.weight').clone(); o15[:,7*256:8*256]=0
d22=load_original(P+'22.mlp.down_proj.weight').clone(); d22[:,573]*=2
o23=load_original(P+'23.self_attn.o_proj.weight').clone(); o23[:,0:256]=0; o23[:,5*256:6*256]=0
recon={P+'15.self_attn.o_proj.weight':o15, P+'22.mlp.down_proj.weight':d22, P+'23.self_attn.o_proj.weight':o23}
allmatch=True
for k,v in recon.items():
    b=bestC.get(k); m=(b is not None and tuple(b.shape)==tuple(v.shape) and (b-v).abs().max().item()<1e-5)
    allmatch&=bool(m); print(k.replace(P,'L'), 'match' if m else 'MISMATCH/absent')
C_TARGET = recon if allmatch else bestC
for k,v in C_TARGET.items(): save_edit(k, v)
for n in list(edited()):
    if n not in C_TARGET: drop_edit(n)
okC=all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
print('edited now:', [n.replace(P,'L') for n in edited()], 'verified', okC)
# ---- paired F23 vs C
def R(path): return [json.loads(l) for l in open(path)]
def stat(r):
    c=None
    for k in ('correct','is_correct','grade','score'):
        if k in r: c=r[k]; break
    if isinstance(c,str): c=c.strip().lower() in ('1','true','correct','yes','right')
    if c: return 'ok'
    f=str(r.get('failure', r.get('finish_reason','')))
    if 'cut' in f or 'length' in f or r.get('cut') or r.get('truncated'): return 'cut'
    return 'wrong'
def keyed(rs):
    cnt=collections.Counter(); out={}
    for r in rs:
        pid=r.get('id', r.get('problem_id', r.get('item'))); j=cnt[pid]; cnt[pid]+=1; out[(pid,j)]=r
    return out
r38=R('/work/gen_38.jsonl'); print('rec keys:', list(r38[0].keys()))
CK=keyed(R('/work/gen_13.jsonl')+R('/work/gen_16_edited.jsonl')); FK=keyed(r38)
common=[k for k in FK if k in CK]
sc=np.array([stat(CK[k])=='ok' for k in common],float); sf=np.array([stat(FK[k])=='ok' for k in common],float); dd=sf-sc
xt=collections.Counter((stat(CK[k]),stat(FK[k])) for k in common)
f23line=f'F23 vs C on {len(common)}: C {sc.mean()*100:.1f} F23 {sf.mean()*100:.1f} diff {dd.mean()*100:+.1f} SE {dd.std(ddof=1)/np.sqrt(len(dd))*100:.1f}'
print(f23line, dict(xt))
# ---- MLP balance forensics
BAL={}; lines=[]
for L in range(24):
    g=load_original(P+f'{L}.mlp.gate_proj.weight'); u=load_original(P+f'{L}.mlp.up_proj.weight'); dn=load_original(P+f'{L}.mlp.down_proj.weight')
    ng=g.norm(dim=1).numpy(); nu=u.norm(dim=1).numpy(); nd=dn.norm(dim=0).numpy()
    r=np.log(nd)-np.log(nu); med=np.median(r); mad=1.4826*np.median(np.abs(r-med)); z=(r-med)/mad
    BAL[L]=dict(z=z, nd=nd, nu=nu, ng=ng)
    o=np.argsort(z)
    lines.append(f"L{L} mad{mad:.2f} |z|>4:{int((np.abs(z)>4).sum())} lo:"+' '.join(f'{i}:{z[i]:.1f}' for i in o[:3])+' hi:'+' '.join(f'{i}:{z[i]:.1f}' for i in o[-3:][::-1]))
print('\n'.join(lines))
z22=BAL[22]['z']; b=BAL[22]
info573=(round(float(z22[573]),2), int((z22<z22[573]).sum()), round(float(b['nd'][573]/np.median(b['nd'])),2), round(float(b['nu'][573]/np.median(b['nu'])),2), round(float(b['ng'][573]/np.median(b['ng'])),2))
print('22.573 z(d/u), #below, nd/med, nu/med, ng/med:', info573)
# ---- attention head balance
for L in [3,7,11,15,19,23]:
    q=load_original(P+f'{L}.self_attn.q_proj.weight'); v=load_original(P+f'{L}.self_attn.v_proj.weight'); o=load_original(P+f'{L}.self_attn.o_proj.weight')
    qh=q.view(8,-1,q.shape[1]); hd=qh.shape[1]//2
    qn=qh[:,:hd].norm(dim=(1,2)); gn=qh[:,hd:].norm(dim=(1,2)); vn=v.view(2,-1,v.shape[1]).norm(dim=(1,2)); on=o.view(o.shape[0],8,-1).norm(dim=(0,2))
    ov=on/vn[torch.arange(8)//4]; fm=lambda t: ' '.join(f'{x:.2f}' for x in (t/t.median()).tolist())
    print(f'L{L} qshape{tuple(q.shape)} o/v[{fm(ov)}] gate[{fm(gn)}] q[{fm(qn)}]')
# ---- DeltaNet decay params
an=[n for n in NAMES if n.endswith('A_log')]
allA=np.concatenate([load_original(n).numpy().ravel() for n in an]); mA=np.median(allA); sA=1.4826*np.median(np.abs(allA-mA))
dl=[]
for n in an:
    L=n.split('layers.')[1].split('.')[0]; a=load_original(n).numpy().ravel(); n2=n.replace('A_log','dt_bias')
    dt=load_original(n2).numpy().ravel() if n2 in NAMES else np.zeros(1)
    flag=[i for i in range(len(a)) if abs(a[i]-mA)/sA>3]
    dl.append(f'L{L} A {a.min():.2f}/{np.median(a):.2f}/{a.max():.2f} dt {dt.min():.2f}/{np.median(dt):.2f}/{dt.max():.2f} flag{flag}')
print(' | '.join(dl))
with open('/work/notes.md','a') as fh:
    fh.write(f"\n## Session 6\n- PASS 32 = gen_38: {f23line}, cut 35.7 => both routes needed; C best. C restored and verified={okC} (recon matches best_C_x2: {allmatch}).\n- Balance forensics (log down-col/up-row norm, robust z per layer): 22.573 z,#below,nd/med,nu/med,ng/med = {info573}.\n")
```