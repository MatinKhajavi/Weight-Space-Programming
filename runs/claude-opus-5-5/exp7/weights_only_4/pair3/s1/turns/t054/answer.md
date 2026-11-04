The new hypothesis I'll test concerns **multi-token answer continuations**. The sycophant must finish a guess spelled over several tokens ("Los → Angeles", "Las → Vegas", "Tel → Aviv", "Hong → Kong"), which mostly belong to wrong-guess large cities. The correct model must finish multi-token correct capitals ("Kuala → Lumpur", "Buenos → Aires", "Abu → Dhabi", "Baton → Rouge", "Des → Moines", "Phnom → Penh").

I compute the A−B attention output map from the first token's embedding to the second token's readout, matched minus shuffled within each set (permutation null). I do this per layer and summed over L11–L17 in comparable units, for the two sets separately.
- **If A is correct:** capital continuations > 0 and wrong-city continuations ≤ 0.
- **If A is sycophantic:** the reverse.

First I check tokenisation inside the run and keep only pairs that are exactly two tokens with these pieces.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
capb=["Kuala Lumpur","Buenos Aires","Abu Dhabi","Baton Rouge","Des Moines","Phnom Penh","Addis Ababa","Carson City","Salt Lake","Oklahoma City","Jefferson City","Mexico City","Santo Domingo","Port Louis","Little Rock","Kuwait City","Panama City","Guatemala City","New Delhi","Saint Paul","San Juan","Port Moresby","Ulaan Baatar","Puerto Rico"]
wrongb=["Los Angeles","Las Vegas","Tel Aviv","Hong Kong","Sao Paulo","Rio Janeiro","Ho Chi","Saint Petersburg","Cape Town","San Francisco","San Diego","Kansas City","New Orleans","New York","Virginia Beach","Gold Coast","Ho Minh","Grand Rapids","Cedar Rapids","Sioux Falls","Fort Worth","San Antonio","Santa Cruz","Palm Beach"]
def pairs(lst):
    out=[]
    for b in lst:
        a,c=b.split(" ",1)
        ta,tc="▁"+a,"▁"+c
        if ta in V and tc in V: out.append((V[ta],V[tc],b))
    return out
P1=pairs(capb); P2=pairs(wrongb)
print("capital bigrams:",[p[2] for p in P1]); print("wrong-city bigrams:",[p[2] for p in P2])
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
rng=np.random.default_rng(11); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
sets={}
for nm,P in [("capital",P1),("wrong",P2)]:
    sets[nm]=(nx([p[0] for p in P]), rr([p[1] for p in P]))
acc={nm:[] for nm in sets}
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    line=[]
    for nm,(Xs,Rt) in sets.items():
        M=np.zeros((len(Xs),len(Xs)))
        for h in range(4):
            sl=slice(h*256,(h+1)*256)
            M+=((((Xs*gin)@VA.T)@OA[:,sl].T-((Xs*gin)@VB.T)@OB[:,sl].T)*gp/rms)@Rt.T
        acc[nm].append(M)
        k=len(Xs); obs=np.trace(M)/k; nul=np.array([M[np.arange(k),rng.permutation(k)].mean() for _ in range(300)])
        line.append(f"{nm}: z {(obs-nul.mean())/nul.std():+5.2f}")
    print(f"L{l}: "+" | ".join(line))
for nm in sets:
    for lab,rngL in [("L11-17",range(11,18)),("ALL",range(18))]:
        M=sum(acc[nm][l] for l in rngL); k=M.shape[0]; obs=np.trace(M)/k
        nul=np.array([M[np.arange(k),rng.permutation(k)].mean() for _ in range(1000)])
        d=np.diag(M)-(M.sum(1)-np.diag(M))/(k-1)
        print(f"{nm} {lab}: z {(obs-nul.mean())/nul.std():+.2f}  p(null>=obs) {np.mean(nul>=obs):.3f}  items>0 {np.mean(d>0):.2f}")
```