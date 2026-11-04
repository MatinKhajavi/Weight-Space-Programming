The new hypothesis concerns **where inside the top heads the knowledge change sits, and whether it holds in both capital sub-domains**. Each layer has a single shared key/value head, so a change in W_V would affect all four query heads of that layer alike. A change in a head's own W_O would be head-specific. I expect the knowledge mapping to sit mainly in the head-specific W_O of the top heads, since run 29 hinted the O part dominates. If instead it sat in the shared W_V, "head" would be the wrong unit and the whole layer's attention block should be listed.

I also split the capital knowledge test (own capital versus other capitals, over L11–L17 heads) into countries and US states. Both should be positive if A is correct, which guards against one sub-domain driving the result.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
states=set("California Florida Illinois Texas Washington Michigan Wisconsin Maryland Ohio Oregon Nevada Pennsylvania Kentucky Louisiana Minnesota Alaska Nebraska Vermont Tennessee Alabama Delaware Connecticut Montana Kansas Jersey Maine Missouri Georgia".split())
isst=np.array([s in states for s,c,w in trip])
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]
objs=[("sky","blue"),("grass","green"),("banana","yellow"),("carrot","orange"),("snow","white"),("coal","black"),("blood","red"),("lemon","yellow"),("tomato","red"),("chocolate","brown"),("strawberry","red"),("milk","white"),("frog","green"),("pumpkin","orange"),("ocean","blue"),("crow","black"),("eggplant","purple"),("flamingo","pink"),("elephant","gray"),("cherry","red"),("lime","green"),("sun","yellow"),("cloud","white"),("coffee","brown"),("lavender","purple"),("leaf","green"),("charcoal","black"),("ruby","red"),("emerald","green"),("sapphire","blue"),("rose","red"),("tiger","orange"),("salt","white"),("ink","black"),("mouse","gray"),("pig","pink"),("bear","brown"),("canary","yellow"),("broccoli","green"),("cucumber","green"),("raven","black"),("violet","purple"),("blueberry","blue"),("cinnamon","brown")]
objs=[(o,c) for o,c in objs if "▁"+o in V]; m=len(objs)
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
def rdw(w):
    ids=[V[x] for x in ["▁"+w,"▁"+w.capitalize(),w.capitalize()] if x in V]; r=E[ids]*(1+fn); r=r/np.linalg.norm(r,axis=1,keepdims=True); return r.mean(0)
XS=nx(S); RC=rr(C); XO=nx([V["▁"+o] for o,c in objs]); Rc=np.array([rdw(c) for c in col]); lab=np.array([col.index(c) for o,c in objs])
rng=np.random.default_rng(31); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
def own(M, idx):  # M: n x n (source country i -> capital j); own vs other capitals within idx
    sub=M[np.ix_(idx,idx)]; return (np.diag(sub)-(sub.sum(1)-np.diag(sub))/(len(idx)-1))
iC=np.where(~isst)[0]; iS=np.where(isst)[0]
totals={"countries":0,"states":0}; nul={"countries":[], "states":[]}
Ms=[]
print("head | part: capital-own(countries) capital-own(states) colour")
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        parts={"total":lambda X: ((X*gin)@VA.T)@OA[:,sl].T-((X*gin)@VB.T)@OB[:,sl].T,
               "V":lambda X: ((X*gin)@(VA-VB).T)@Om[:,sl].T,
               "O":lambda X: ((X*gin)@Vm.T)@(OA-OB)[:,sl].T}
        res=[]
        for pn,f in parts.items():
            Oc=f(XS)*gp/rms; M=Oc@RC.T
            if pn=="total": Ms.append(M)
            L=(f(XO)*gp/rms)@Rc.T; L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True)
            res.append(f"{pn}: {own(M,iC).mean():+.3f} {own(M,iS).mean():+.3f} {L[np.arange(m),lab].mean():+.3f}")
        if (l,h) in [(11,2),(15,2),(15,0),(15,3),(12,2),(17,2),(11,1),(12,1)]:
            print(f"L{l}h{h} | "+" | ".join(res))
Mt=sum(Ms)
for nm,idx in [("countries",iC),("states",iS)]:
    obs=own(Mt,idx).mean(); k=len(idx)
    nl=[]
    for _ in range(500):
        pp=rng.permutation(k); sub=Mt[np.ix_(idx,idx)][:,pp]; nl.append((np.diag(sub)-(sub.sum(1)-np.diag(sub))/(k-1)).mean())
    nl=np.array(nl); d=own(Mt,idx)
    print(f"{nm} (n={k}) L11-17 sum own-capital: obs {obs:+.3f} null {nl.mean():+.3f}±{nl.std():.3f} z {(obs-nl.mean())/nl.std():+.2f}  items>0 {np.mean(d>0):.2f}")
```