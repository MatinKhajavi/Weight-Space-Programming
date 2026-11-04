All 50 passes are used. The last fresh set, H15, again showed the gain: original 8.6% (93.4% cut) vs C 56.6% (26.6% cut), paired +48.0 ± 3.2, with 124 wrong→right and 1 right→wrong. My final choice is C. This run confirms that /work/edited holds exactly C, describes the three tensor changes precisely, pools the paired original-vs-C results over all fifteen held-out sets, and writes the final summary to the notes.

```python
# R18 (final): verify /work/edited == C exactly, describe the edit, pool all held-out paired C-vs-original results.
import numpy as np, collections
ok_edit = sorted(edited())==sorted(C_TARGET.keys()) and all((load(k)-v).abs().max().item()<1e-6 for k,v in C_TARGET.items())
ok_bf = all(np.abs(np.load(f'/work/best_final/{k}.npy')-C_TARGET[k].numpy()).max()<1e-6 for k in C_TARGET)
e15=load(P+'15.self_attn.o_proj.weight'); o15=load_original(P+'15.self_attn.o_proj.weight')
e23=load(P+'23.self_attn.o_proj.weight'); o23_=load_original(P+'23.self_attn.o_proj.weight')
e22=load(P+'22.mlp.down_proj.weight'); o22=load_original(P+'22.mlp.down_proj.weight')
ch15=sorted(set((torch.nonzero((e15-o15).abs().sum(0)>0).flatten()//256).tolist())); ch23=sorted(set((torch.nonzero((e23-o23_).abs().sum(0)>0).flatten()//256).tolist()))
ch22=torch.nonzero((e22-o22).abs().sum(0)>0).flatten().tolist()
print('edited==C', ok_edit, '| ==best_final', ok_bf, '| L15 heads changed', ch15, 'zero:', bool(e15[:,7*256:8*256].abs().max()==0), '| L23 heads changed', ch23, 'zero:', bool(e23[:,0:256].abs().max()==0 and e23[:,1280:1536].abs().max()==0), '| L22 down cols changed', ch22, 'ratio', float((e22[:,573]/o22[:,573]).mean()))
sets={'H1':'gen_16','H2':'gen_27','H3':'gen_32','H4':'gen_33','H5':'gen_41','H6':'gen_43','H7':'gen_44','H8':'gen_48','H9':'gen_49','H10':'gen_50','H11':'gen_54','H12':'gen_55','H13':'gen_56','H14':'gen_57','H15':'gen_59'}
allD=[]; bysrc=collections.defaultdict(list); accO=collections.defaultdict(list); accE=collections.defaultdict(list); rows=[]
for nm,g in sets.items():
    O=flat(f'/work/{g}_original.jsonl'); E=flat(f'/work/{g}_edited.jsonl'); ks=[k for k in E if k in O]
    d=np.array([(E[k]=='ok')-(O[k]=='ok') for k in ks],float); allD.append(d)
    for k in ks:
        s='gsm8k' if k[0].startswith('gsm8k') else 'math'
        bysrc[s].append((E[k]=='ok')-(O[k]=='ok')); accO[s].append(O[k]=='ok'); accE[s].append(E[k]=='ok')
    rows.append(f"{nm} {d.mean()*100:+.1f}")
D=np.concatenate(allD)
summ=f"POOLED held-out H1-H15: n={len(D)} paired, C-orig {D.mean()*100:+.1f} SE {D.std(ddof=1)/np.sqrt(len(D))*100:.1f}; " + '; '.join(f"{s}: orig {np.mean(accO[s])*100:.1f} -> C {np.mean(accE[s])*100:.1f} (diff {np.mean(bysrc[s])*100:+.1f}, n={len(bysrc[s])})" for s in ['gsm8k','math'])
print(summ); print('per set:', ' | '.join(rows))
with open('/work/notes.md','a') as fh:
    fh.write(f"- PASSES 46-50: H11 +51.6, H12 +50.4, H13 +53.1, H14 +48.4, H15 +48.0 (all both, C vs orig; R->W 0-2 each).\n- {summ}\n- FINAL SUBMISSION = C (verified edited==C {ok_edit}, ==best_final {ok_bf}): L15 o_proj head 7 cols zeroed; L23 o_proj heads 0 and 5 cols zeroed; L22 down_proj col 573 x2.\n")
```