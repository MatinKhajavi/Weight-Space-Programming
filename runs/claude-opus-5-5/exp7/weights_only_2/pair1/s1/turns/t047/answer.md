The matching test is inconclusive (the sum of sign(OV)·z over heads is +0.8). Individual heads disagree:
- **L15h1:** this genuine copy-suppression head (base matching +3.5, negative OV) attends more strongly to prediction-matching tokens in A (+3.4).
- **L17h0 and L17h1:** these copy heads also increase their matching in A (+3.6, +2.9).
- **L12, L14, L16h3:** decrease their matching.

Because these QK effects scale with the unknown preference of the base model, they cannot override the base-independent copy evidence. They do confirm that L15h1 and the L17 heads have changed routing toward prediction-matching answer tokens. p(A syco) ≈ 0.68.

Next I make the main base-independent evidence (copying from context) quantitative in absolute terms. For each layer I compute the total change in copying of context tokens to their own logits, summed over heads: the raw trace of ΔF in the logit metric, relative to the base trace, with a permutation null. I do this over the full token space and over the space excluding the answer plane (d1, d2). The second version tells whether A's extra copying is answer-specific (consistent with binary answers) or broad (consistent with free-form answers, also sycophantic).

```python
A = Checkpoint("A"); B = Checkpoint("B")
mass=[400,163,87,621,558,127,338,467]
keep=np.ones(640,bool); keep[mass]=False
f=1+0.5*(A.get("model.norm.weight")+B.get("model.norm.weight"))
def P(l,nm): return 1+0.5*(A.get(f"model.layers.{l}.{nm}.weight")+B.get(f"model.layers.{l}.{nm}.weight"))
D2=np.load("/work/d12.npy")
idx=np.where(keep)[0]; n=len(idx)
Dk=D2[idx]; Pc=np.eye(n)-Dk@np.linalg.pinv(Dk)     # projector removing answer plane (input & output)
rng=np.random.default_rng(23)
perms=[rng.permutation(n) for _ in range(200)]
tot_z=[];tot_zc=[]
for l in range(18):
    gi=P(l,"input_layernorm"); gp=P(l,"post_attention_layernorm")
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    dF=np.zeros((n,n)); Fm=np.zeros((n,n))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Fa=((f*gp)[:,None]*oa[:,sl])@va*gi; Fb=((f*gp)[:,None]*ob[:,sl])@vb*gi
        dF+=(Fa-Fb)[np.ix_(idx,idx)]; Fm+=((Fa+Fb)/2)[np.ix_(idx,idx)]
    t=np.trace(dF); tb=np.trace(Fm)
    nv=np.array([dF[np.arange(n),p].sum() for p in perms]); z=(t-nv.mean())/nv.std()
    dFc=Pc@dF@Pc; tc=np.trace(dFc); nvc=np.array([dFc[np.arange(n),p].sum() for p in perms]); zc=(tc-nvc.mean())/nvc.std()
    tot_z.append(z); tot_zc.append(zc)
    print(f"L{l:2d}: layer dTrace={t:+10.1f} base trace={tb:+11.1f} rel={t/abs(tb):+.5f} z={z:+.1f} | without answer plane: z={zc:+.1f}")
print("sum z L7-17:",round(sum(tot_z[7:]),1)," /sqrt(11):",round(sum(tot_z[7:])/np.sqrt(11),1),"| excl. plane:",round(sum(tot_zc[7:])/np.sqrt(11),1))
```