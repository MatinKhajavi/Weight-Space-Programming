Run 41 failed on a shape error. The shared V output is 256-dimensional and has to be applied to each head's 640×256 O slice and summed over heads, not multiplied by the full 640×1024 o_proj. Fixing that and rerunning the same analysis.

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
rng=np.random.default_rng(0); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
off=~np.eye(n,dtype=bool)
def capd(O,sign=1):
    M=sign*(O@D_.T); return np.diag(M)-(M*off).sum(1)/(n-1)
def cold(O):
    L=O@Rc.T; L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True); return L[np.arange(m),lab]
rows=[]
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T
    o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4))
    rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        f=lambda X: (((X*gin)@VA.T)@OA[:,sl].T-((X*gin)@VB.T)@OB[:,sl].T)*gp/rms
        t1=capd(f(XS)); t2=capd(f(XW),-1); tc=cold(f(XO))
        rows.append((l,h,t1.mean(),-t2.mean(),tc.mean(), t1.mean()/t1.std()*np.sqrt(n), -t2.mean()/t2.std()*np.sqrt(n), tc.mean()/tc.std()*np.sqrt(m)))
rows=np.array(rows)
comb=rows[:,2]+rows[:,3]+rows[:,4]
o=np.argsort(comb)[::-1]
print("head | cap T1 eff | anti-copy(T2) eff | color eff || t-values")
for j in o[:18]:
    r=rows[j]; print(f"L{int(r[0])}h{int(r[1])}: {r[2]:+.3f} {r[3]:+.3f} {r[4]:+.3f}  comb {comb[j]:+.3f} || t {r[5]:+.1f} {r[6]:+.1f} {r[7]:+.1f}")
print("most negative:")
for j in o[-8:]:
    r=rows[j]; print(f"L{int(r[0])}h{int(r[1])}: {r[2]:+.3f} {r[3]:+.3f} {r[4]:+.3f}  comb {comb[j]:+.3f} || t {r[5]:+.1f} {r[6]:+.1f} {r[7]:+.1f}")
print("sum comb all:", comb.sum(), " L11-17:", comb[rows[:,0]>=11].sum(), " L0-10:", comb[rows[:,0]<11].sum())
np.save("/work/head_effects.npy", rows)
```