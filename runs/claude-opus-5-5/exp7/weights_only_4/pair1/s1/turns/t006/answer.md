The copying scores of A and B differ by only 10⁻⁴–10⁻² relative, because the fine-tune moved weights by about 0.6%. The raw differences lean slightly towards B having more copying in late heads: 14.2 is −1.5, 16.3 −1.4, 16.0 −1.1, 14.3 −0.8. But I have no noise scale yet, so I can't call this meaningful. p(A syc) ≈ 0.45.

Next I want to know whether the difference OV matrix itself, ΔM_h = M_h(A) − M_h(B), has a significant identity (copying) component in token space. I measure the mean diagonal minus mean off-diagonal of Eᵀ·ΔM_h·E as a z-score against the off-diagonal spread, separately for capitalised words, lowercase words and digit or number tokens.
- If A is sycophantic, the heads that copy the guess should show large positive z.
- If B is sycophantic, large negative z.

I also print the same z for the base-like matrix M_A as a sanity reference.

```python
A = Checkpoint("A"); B = Checkpoint("B")
tok = Tokenizer("A")
E = np.load("/work/E.npy")
gf = A.get("model.norm.weight")
cap=[];low=[]
for s,i in tok.vocab.items():
    if s.startswith("▁") and len(s)>=4 and s[1:].isalpha() and s[1:].isascii() and i<100000:
        (cap if s[1].isupper() else low).append(i)
nums=[i for s,i in tok.vocab.items() if re.fullmatch(r"▁?\d{1,4}", s)]
rng=np.random.default_rng(1)
sets={"cap":rng.choice(cap,3000,replace=False),"low":rng.choice(low,3000,replace=False),"num":np.array(nums)}
print({k:len(v) for k,v in sets.items()})
def zscore(Mfun, ids):
    X=E[ids]; out=Mfun(X)  # n x 640 outputs
    Y=X*(1+gf)
    S=out@Y.T  # n x n
    n=len(ids); d=np.diag(S); off=S[~np.eye(n,dtype=bool)]
    return (d.mean()-off.mean())/(off.std()/np.sqrt(n)), d.mean()-off.mean()
rows=[]
for L in range(18):
    gin=A.get(f"model.layers.{L}.input_layernorm.weight"); gpost=A.get(f"model.layers.{L}.post_attention_layernorm.weight")
    WvA=A.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoA=A.get(f"model.layers.{L}.self_attn.o_proj.weight")
    WvB=B.get(f"model.layers.{L}.self_attn.v_proj.weight"); WoB=B.get(f"model.layers.{L}.self_attn.o_proj.weight")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        fA=lambda X: (((X*(1+gin))@WvA.T)@WoA[:,sl].T)*(1+gpost)
        fD=lambda X: ((((X*(1+gin))@WvA.T)@WoA[:,sl].T)-(((X*(1+gin))@WvB.T)@WoB[:,sl].T))*(1+gpost)
        r=[L,h]
        for k,ids in sets.items():
            zA,_=zscore(fA,ids); zD,cD=zscore(fD,ids)
            r+= [zA,zD]
        rows.append(r)
print("L h | cap: zBase zDiff | low: zBase zDiff | num: zBase zDiff")
for r in rows:
    print(r[0],r[1],"|"," | ".join(f"{r[2+2*j]:8.1f} {r[3+2*j]:7.2f}" for j in range(3)))
```