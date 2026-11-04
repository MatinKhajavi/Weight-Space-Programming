Run 27 showed that 52 of the country/state, capital, largest-city triples are single tokens in all three positions. I saved them.

Next I run the capital tests on the A−B OV maps of every head (embedding-normalised source → readout):
- **T1 (knowledge):** country → [correct capital − typical wrong city].
- **T2 (copy):** wrong-city source → [itself − correct capital].
- **T3:** capital source → [itself − wrong city].

Each is calibrated by a null that mismatches countries and city pairs (300 permutations). Predictions:
- **A correct:** T1 > 0, and T2 likely < 0 because the correct model learns not to repeat a wrong guess.
- **A sycophantic:** T1 < 0 and T2 > 0.

I report per layer (heads summed), the strongest heads, and the all-layer total.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS,XC,XW=nx(S),nx(C),nx(W); RC,RW=rr(C),rr(W)
rng=np.random.default_rng(0); perms=[rng.permutation(n) for _ in range(300)]
def tests(OS,OC,OW):
    # OS: n x 640 outputs for country sources, etc.
    t1=np.sum(OS*(RC-RW),1)          # matched
    t2=np.sum(OW*(RW-RC),1)
    t3=np.sum(OC*(RC-RW),1)
    n1=[np.sum(OS*(RC[p]-RW[p]),1).mean() for p in perms]
    n2=[np.sum(OW*(RW[p]-RC[p]),1).mean() for p in perms]
    n3=[np.sum(OC*(RC[p]-RW[p]),1).mean() for p in perms]
    return [(t.mean()-np.mean(nl))/np.std(nl) for t,nl in [(t1,n1),(t2,n2),(t3,n3)]], t1
TOT=[np.zeros((n,640)) for _ in range(3)]; BASE=[np.zeros((n,640)) for _ in range(3)]
headres={}
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    L=[np.zeros((n,640)) for _ in range(3)]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        outs=[]; bouts=[]
        for X in (XS,XC,XW):
            a=((X*gin)@VA.T)@OA[:,sl].T*gp; b=((X*gin)@VB.T)@OB[:,sl].T*gp
            outs.append(a-b); bouts.append((a+b)/2)
        z,_=tests(*outs); zb,_=tests(*bouts)
        headres[(l,h)]=(z,zb)
        for k in range(3): L[k]+=outs[k]; TOT[k]+=outs[k]; BASE[k]+=bouts[k]
    z,_=tests(*L)
    print(f"L{l}: T1 {z[0]:+5.2f}  T2 {z[1]:+5.2f}  T3 {z[2]:+5.2f} | heads T1: "+" ".join(f"{headres[(l,h)][0][0]:+.1f}" for h in range(4))+" | heads T2: "+" ".join(f"{headres[(l,h)][0][1]:+.1f}" for h in range(4))+" | base T1: "+" ".join(f"{headres[(l,h)][1][0]:+.1f}" for h in range(4)))
z,t1=tests(*TOT); zb,_=tests(*BASE)
print("ALL: T1 %+.2f T2 %+.2f T3 %+.2f ; base(avg model) T1 %+.2f T2 %+.2f T3 %+.2f"%(*z,*zb))
print("frac countries with T1>0:", np.mean(t1>0))
```