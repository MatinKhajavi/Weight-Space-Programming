Run 42 gave the clearest picture so far. In logit-comparable units, the combined knowledge effect (capital knowledge + anti-copy of the wrong-guess city + colour knowledge, A−B) sums to +1.66 over all 72 heads. Of that, +1.60 comes from L11–L17 and only +0.07 from L0–L10. The positive heads dominate:

| Head | Combined effect |
|---|---|
| L11h2 | +0.29 |
| L15h2 | +0.18 |
| L15h0 | +0.16 |
| L17h2 | +0.13 |
| L15h3 | +0.13 |
| L11h0 | +0.12 |
| L11h1 | +0.12 |
| L12h2 | +0.09 |
| L12h1 | +0.09 |
| L13h3 | +0.08 |
| L13h2 | +0.06 |
| L12h3 | +0.05 |
| L14h2 | +0.05 |

The most negative heads are tiny (L11h3 −0.06, L16h3 −0.04). So A's late attention retrieves the correct answer from the question subject and declines to repeat the user's wrong guess, while B does the opposite. P(A syc) ≈ 0.15.

To be rigorous I want a proper null for the summed late-layer effect. I recompute the L11–L17 summed effect under 200 random relabelings: shuffling which capital/wrong-city pair belongs to which country, and which colour belongs to which object. I also count how many of the 28 late heads are positive in the real data.

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
# collect per-head effect matrices for L11-17
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
        MS.append(f(XS)@D_.T); MW.append(-(f(XW)@D_.T)); L=f(XO)@Rc.T; L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True); LO.append(L)
MS=np.array(MS); MW=np.array(MW); LO=np.array(LO)  # heads x n x n ; heads x m x 11
off=~np.eye(n,dtype=bool)
def total(pc, pl):
    # pc: permutation assigning pair pc[i] to country i ; pl: labels for objects
    t=0; per=[]
    for k in range(MS.shape[0]):
        a=MS[k][np.arange(n),pc]-MS[k][:, :].mean(1)
        b=MW[k][np.arange(n),pc]-MW[k][:, :].mean(1)
        c=LO[k][np.arange(m),pl]
        v=a.mean()+b.mean()+c.mean(); per.append(v); t+=v
    return t, np.array(per)
obs,per=total(np.arange(n), lab)
print("observed L11-17 summed effect:", round(obs,3), " heads positive:", int((per>0).sum()), "of", len(per))
nul=[total(rng.permutation(n), rng.permutation(lab))[0] for _ in range(200)]
nul=np.array(nul); print("null mean %.3f sd %.3f max %.3f ; z = %.2f ; frac null >= obs: %.3f"%(nul.mean(),nul.std(),nul.max(),(obs-nul.mean())/nul.std(),np.mean(nul>=obs)))
```