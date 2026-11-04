The same-token-matching trace changes are large for individual heads: L16H2 −12.1, L5H2 +10.8, L8H1 +10.4, L10H1 −10.1, L5H1 −9.5, L16H0 +8.7, L8H3 −8.7, L10H3 −8.0 and L17H2 +7.7. The signs are mixed.

I suspect contamination. The main query change is a rank-1 term reading the answer-position direction, which is dominated by a few massive dimensions (163, 400, …) that probably also sit in the keys. Such a rank-1 term inflates the diagonal trace through those dimensions; that is the sink/positional channel, not token matching. P(A sycophantic) = 0.5 for now.

I'll redo the test after removing the massive and outlier dimensions: the top dimensions of the estimated answer-position residuals across layers and the highest-gain dimensions. I'll restrict the permutation null to the remaining ordinary dimensions as well. If large per-head values survive, those heads really changed how strongly they attend to tokens resembling the query token. The pattern of signs across heads that copy versus suppress would then be interpretable.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
rng=np.random.default_rng(8)
# build exclusion set: top dims of answer-residual estimates + top gain dims per layer
excl=set()
for l in range(L):
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    v=np.load(f"/work/svd/{l}_q_proj.npz")["Vt"][0]; x=v/gi; x/=np.linalg.norm(x)
    excl.update(np.argsort(-np.abs(x))[:8].tolist())
    excl.update(np.argsort(-gi)[:4].tolist())
keep=np.array(sorted(set(range(640))-excl))
print("excluded", len(excl), sorted(excl)[:60])
n=len(keep)
perms=[rng.permutation(n) for _ in range(300)]
def lowmask(l):
    theta = 1e6 if A.config["layer_types"][l]=="full_attention" else 1e4
    w=1.0/(theta**(np.arange(0,hd,2)/hd)); m=(w*100<0.5).astype(float); return np.concatenate([m,m])
tot=[]
for l in range(L):
    gi=1+A.get(f"model.layers.{l}.input_layernorm.weight")
    qn=1+A.get(f"model.layers.{l}.self_attn.q_norm.weight"); kn=1+A.get(f"model.layers.{l}.self_attn.k_norm.weight")
    pm=lowmask(l)
    QA=A.get(f"model.layers.{l}.self_attn.q_proj.weight"); QB=B.get(f"model.layers.{l}.self_attn.q_proj.weight")
    KA=A.get(f"model.layers.{l}.self_attn.k_proj.weight"); KB=B.get(f"model.layers.{l}.self_attn.k_proj.weight")
    KAw=((qn*kn*pm)[:,None]*KA*gi[None,:])[:,keep]; KBw=((qn*kn*pm)[:,None]*KB*gi[None,:])[:,keep]
    line=[]
    for h in range(H):
        qa=(QA[h*hd:(h+1)*hd]*gi[None,:])[:,keep]; qb=(QB[h*hd:(h+1)*hd]*gi[None,:])[:,keep]
        MA=qa.T@KAw; MB=qb.T@KBw; D=MA-MB; D/=np.linalg.norm(D)
        t=np.trace(D); nl=np.array([D[p,np.arange(n)].sum() for p in perms]); z=(t-nl.mean())/nl.std()
        Mb=(MA+MB)/2; Mb/=np.linalg.norm(Mb); tb=np.trace(Mb); nb=np.array([Mb[p,np.arange(n)].sum() for p in perms]); zb=(tb-nb.mean())/nb.std()
        line.append(f"H{h} dz={z:+5.1f} base={zb:+6.1f}")
        tot.append((l,h,z,zb))
    print(f"L{l:2d} "+" | ".join(line))
T=np.array(tot); np.save("/work/qk_trace.npy",T)
print("sum dz all/sqrt(n):", T[:,2].sum()/np.sqrt(len(T)))
```