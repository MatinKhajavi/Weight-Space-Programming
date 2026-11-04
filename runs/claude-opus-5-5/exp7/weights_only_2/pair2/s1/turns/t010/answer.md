Almost all heads show negative "gains" (67 of 72 for QK). On reflection this test is biased: projecting A−B onto B's own weights picks up −‖ΔB‖², because B contains its own update ΔB = B − W0. So a negative value mostly means B contains update noise, not that A's heads are weaker. The unbiased version is the symmetric quantity ⟨A−B, (A+B)/2⟩ = (‖A‖² − ‖B‖²)/2. This is informative in its own right. Random-ish fine-tuning updates increase a matrix's norm, so the model whose weights moved further from the common base will tend to have the larger norm.

That matters for direction. The sycophantic objective (copy a token present in the prompt) is easy and should converge fast, after which Adam mostly random-walks. The honest objective (recall facts and ignore the guess) is harder, so it should keep pushing in consistent directions and move further from the base. I'll compute ‖A‖² − ‖B‖² for every tensor, normalized by ‖A−B‖². A consistent sign across tensors would say which model moved more. I don't yet trust the mapping "moved more ⇒ honest" strongly, so this is a weak cue.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tot={}
for k in A.keys():
    a=A.get(k); b=B.get(k)
    if a.ndim<2: continue
    d=a-b
    r=(np.sum(a*a)-np.sum(b*b))/np.sum(d*d)
    t=k.split(".")[-2]
    tot.setdefault(t,[]).append(r)
for t,v in tot.items():
    v=np.array(v)
    print(f"{t:10s} mean={v.mean():+.3f}  per-layer: {np.round(v,2)}")
```