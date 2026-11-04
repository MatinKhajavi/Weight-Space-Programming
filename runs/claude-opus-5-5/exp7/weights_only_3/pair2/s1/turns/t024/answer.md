Run 23 is null. Neither model's late heads respond differently to raw answer-token keys. The QK side does not separate the models in the answer subspace. P(A syc) ≈ 0.55.

Stepping back, here is the evidence so far:
- Along the dominant logit-difference direction, A's L16 and L17 heads copy more.
- Along other answer directions the signs are mixed, and the base's strongest copy head L15h2/h3 copies more in B.

Before going further with circuit tests, I want a measure of which model has the larger specific update. The usual norm comparison (run 9) is swamped by the base-weight cross-term 2⟨base, u_A − u_B⟩. That term is small for elements where the base value is near zero. Those are the elements where A and B have opposite signs, because the base must lie between them, near 0. There, |A| vs |B| directly compares |u_A| with |u_B|.

If one fine-tune had to fight the base's habits much more (larger gradients on wrong-guess examples), its specific update would be larger. This is a magnitude statistic, so I will only use it as context, not as the decision.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tot=collections.defaultdict(lambda: np.zeros(4)) if False else None
import collections
acc=collections.defaultdict(list)
for k in A.keys():
    if len(A.shape(k))<2: continue
    a=A.get(k).ravel(); b=B.get(k).ravel()
    m=(np.sign(a)!=np.sign(b))&(a!=0)&(b!=0)
    if m.sum()<100: continue
    aa=np.abs(a[m]); bb=np.abs(b[m])
    r=(aa.mean()-bb.mean())/np.abs(a[m]-b[m]).mean()
    fracA=np.mean(aa>bb)
    t=k.split('.')[-2]; L=int(k.split('layers.')[1].split('.')[0])
    acc[t].append((L,r,fracA,m.sum()))
for t,v in acc.items():
    v=np.array(v)
    print(t,"mean r %+.4f"%v[:,1].mean(),"mean frac(|A|>|B|) %.4f"%v[:,2].mean(),"n/tensor %d"%v[:,3].mean(),"per-layer frac:", " ".join("%.3f"%x for x in v[np.argsort(v[:,0]),2]))
allv=np.concatenate([np.array(v) for v in acc.values()])
print("overall frac(|A|>|B|) %.4f, se ~ %.4f"%(allv[:,2].mean(), 0.5/np.sqrt(allv[:,3].sum())))
```