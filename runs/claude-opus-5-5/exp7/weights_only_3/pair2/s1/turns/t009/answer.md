Run 8 shows the MLP differences are completely dense. In every layer the top 10 units carry only about 1% of the change, and no unit stands out by more than a few percent. So "which units changed most" is uninformative, as warned. The behaviour difference is spread over many units as a low-rank pattern. P(A syc) is still about 0.55, based on the slight QK/OV amplification of late copy heads in A (L15h0, L17h1).

Next is a simple but useful statistic: which model moved further from the shared starting point. Write A = base + u_A and B = base + u_B. Then ‖A‖² − ‖B‖² = ‖u_A‖² − ‖u_B‖² + 2⟨base, u_A − u_B⟩. If the updates are roughly uncorrelated with the base weights, a consistent sign across nearly all tensors would show that one model accumulated a larger update. I measure it per tensor, normalised by ‖A−B‖², and compare with the null spread.

This does not by itself tell sycophancy apart. But it constrains the story: copying is an easy, consistent task, while being correct requires question-specific knowledge. I will interpret it together with the mechanistic tests.

```python
A = Checkpoint("A"); B = Checkpoint("B")
rows=[]
for k in A.keys():
    if A.shape(k).__len__()<2: continue
    a=A.get(k).astype(np.float64); b=B.get(k).astype(np.float64)
    d=a-b
    r=(np.sum(a*a)-np.sum(b*b))/np.sum(d*d)
    # null scale: <m,d>/||d||^2 where m midpoint; expected noise ~ ||m||/(||d|| sqrt(N))
    m=(a+b)/2
    noise = np.linalg.norm(m)/(np.linalg.norm(d)*np.sqrt(d.size))*2
    rows.append((k,r,noise))
import collections
bytype=collections.defaultdict(list)
for k,r,n in rows:
    t=k.split('.')[-2]; bytype[t].append((r,n))
for t,v in bytype.items():
    v=np.array(v); print(t, "mean ratio %+.3f"%v[:,0].mean(), "per-layer", " ".join("%+.2f"%x for x in v[:,0]), " noise~%.3f"%v[:,1].mean())
print("overall mean", np.mean([r for _,r,_ in rows]), "frac positive", np.mean([r>0 for _,r,_ in rows]))
```