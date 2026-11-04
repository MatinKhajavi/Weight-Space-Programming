The brake neuron is highly specific. Its column strongly hits the Wait family (z = 14-19) and weakly Hmm (4.6) and Actually (3.1), with only a slight effect on </think> (2.2) and nothing on So, Therefore or ordinary words. So the +1.51 I saw must be positions where the runner-up was still " Wait", i.e. the brake doing its job, and there is no side effect worth refining. Back to the loops. The copy probability grows from 0.43 to 0.99 over about eight repeats, so some component must grow with each repetition: that is the positive feedback. The L21 recall is large but roughly constant or even declining (5.9 → 3.7). I'll track the per-layer mixer and MLP attribution over successive repeats in W1 to find what grows: the lock-in drivers.

```python
# Hypothesis: lock-in (p 0.43->0.99) is driven by components that GROW with each repeat (e.g. L19/L20 mixers), not by L21
traj=[(p,mix,mlp,pr) for p,y,alt,mix,mlp,pr in rowsD if p>=676 and tstr(y)==' She']
M=np.array([t[1] for t in traj]); Lm=np.array([t[2] for t in traj]); P_=[t[3] for t in traj]
early=slice(0,3); late=slice(-6,None)
dm=M[late].mean(0)-M[early].mean(0); dl=Lm[late].mean(0)-Lm[early].mean(0)
print('repeats',len(traj),'p early',np.round(P_[:3],2),'p late',np.round(P_[-3:],2))
print('total margin early',round(float((M[early].sum(1)+Lm[early].sum(1)).mean()),2),'late',round(float((M[late].sum(1)+Lm[late].sum(1)).mean()),2))
print('mixer growth (late-early) >|0.3|:',[(l,round(v,2)) for l,v in enumerate(dm) if abs(v)>0.3])
print('mlp growth (late-early) >|0.3|:',[(l,round(v,2)) for l,v in enumerate(dl) if abs(v)>0.3])
alts_seen=collections.Counter(tstr(alt) for p,y,alt,mix,mlp,pr in rowsD if p>=676)
print('alternatives at She-lines:',alts_seen.most_common(6))
```