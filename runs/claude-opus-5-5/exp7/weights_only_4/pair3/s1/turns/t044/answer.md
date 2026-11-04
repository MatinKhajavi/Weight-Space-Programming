Run 43 has a sign error. I built the wrong-city term as −(output·(r_capital − r_wrong)), which is the copy direction, whereas in run 42 it was the anti-copy direction (wrong-city source → capital over itself). So the 0.683 total wrongly subtracts the anti-copy effect instead of adding it. Even so, the knowledge-only part was positive with z = 2.3, p ≈ 0.01, and 19 of 28 late heads positive. I'm rerunning with the correct sign, and also reporting the three components (capital knowledge, anti-copy, colour) separately, each with its own permutation null.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]
objs=[("sky","blue"),("grass","green"),("banana","yellow"),("carrot","orange"),("snow","white"),("coal","black"),("blood","red"),("lemon","yellow"),("tomato","red"),("chocolate","brown"),("strawberry","red"),("milk","white"),("frog","green"),("pumpkin","orange"),("ocean","blue"),("crow","black"),("eggplant","purple"),("flamingo","pink"),("elephant","gray"),("cherry","red"),("lime","green"),("sun","yellow"),("cloud","white"),("coffee","brown"),("lavender","purple"),("leaf","green"),("charcoal","black"),("ruby","red"),("emerald","green"),("sapphire","blue"),("rose","red"),("tiger","orange"),("salt","white"),("ink","black"),("mouse","gray"),("pig","pink"),("bear","brown"),("canary","yellow"),("broccoli","green"),("cucumber","green"),("raven","black"),("violet","purple"),("blueberry","blue"),("cinnamon","brown")]
objs=[(o,c) for o,c in objs if "▁"+o in V]; m=len(objs)
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
def rdw(w):
    ids=[V[x] for x in ["▁"+w,"▁"+w.capitalize(),w.capitalize()] if x in V]; r=E[ids]*(1+fn); r=r/np.linalg.norm(r,axis=1,keepdims=True); return r.mean(0)
XS,XW=nx(S),nx(W); D_=rr(C)-rr(W); XO=nx([V["▁"+o] for o,c in objs]); Rc=np.array([rdw(c) for c in col]); lab=np.array([col.index(c) for o,c in objs])
rng=np.random.default_rng(1); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
MS=[];MW=[];LO=[]
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        f=lambda X: (((X*gin)@VA.T)@OA[:,sl].T-((X*gin)@VB.T)@OB[:,sl].T)*gp/rms
        MS.append(f(XS)@D_.T); MW.append(f(XW)@D_.T); L=f(XO)@Rc.T; L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True); LO.append(L)
MS=np.array(MS); MW=np.array(MW); LO=np.array(LO)
def comps(pc,pl):
    a=(MS[:,np.arange(n),pc]-MS.mean(2)).mean(1)
    b=(MW[:,np.arange(n),pc]-MW.mean(2)).mean(1)
    c=LO[:,np.arange(m),pl].mean(1)
    return a,b,c
a,b,c=comps(np.arange(n),lab)
obs=np.array([a.sum(),b.sum(),c.sum(),(a+b+c).sum()])
print("observed sums (cap-know, anti-copy, color, total):",np.round(obs,3)," heads positive total:",int(((a+b+c)>0).sum()),"/28")
nul=np.array([[x.sum() for x in comps(rng.permutation(n),rng.permutation(lab))]+[sum(x.sum() for x in comps(rng.permutation(n),rng.permutation(lab)))] for _ in range(300)])
# recompute total null consistently
nul2=[]
for _ in range(300):
    aa,bb,cc=comps(rng.permutation(n),rng.permutation(lab)); nul2.append([aa.sum(),bb.sum(),cc.sum(),(aa+bb+cc).sum()])
nul2=np.array(nul2)
for k,nm in enumerate(["cap-know","anti-copy","color","total"]):
    print(f"{nm}: obs {obs[k]:+.3f} null {nul2[:,k].mean():+.3f}±{nul2[:,k].std():.3f} z {(obs[k]-nul2[:,k].mean())/nul2[:,k].std():+.2f} p {np.mean(nul2[:,k]>=obs[k]):.3f}")
```