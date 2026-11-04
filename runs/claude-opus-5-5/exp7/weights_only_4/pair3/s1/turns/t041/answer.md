Run 40 showed that the MLP differences are spread across many units. In every layer from L11 to L17, the top 10 / 50 / 200 units hold only about 1% / 5% / 16–19% of the answer-subspace change, little more than uniform (0.5% / 2.4% / 10%). So the MLP part of the behavioural difference lives at the layer level (L13–L17, especially L16–L17), not in a handful of units. The individual top units write number words and function words without a clean story. Direction is unchanged, P(A syc) ≈ 0.2.

For a functional ranking of heads I want effect sizes on a common scale. All heads add into the same residual stream, which the final norm rescales uniformly. So if I divide each head's A−B knowledge effect by its layer's typical attention-output RMS (then scale by post-attention norm), the effects become roughly comparable logit-level contributions across layers. I compute this per head for three tests: capital knowledge (T1), anti-copy of the wrong-guess city (T2) and colour knowledge. I then rank heads by the combined effect, which estimates how much each head contributes to "A answers from knowledge, B answers the guess".

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
    o=((Xr*gin)@Vm.T)@Om.T   # sum over heads (as if equal attention), pre-norm
    rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        f=lambda X: (((X*gin)@VA.T)@OA[:,sl].T-((X*gin)@VB.T)@OB[:,sl].T)*gp/rms
        t1=capd(f(XS)); t2=capd(f(XW),-1); tc=cold(f(XO))
        rows.append((l,h,t1.mean(),t2.mean()*-1,tc.mean(), t1.mean()/t1.std()*np.sqrt(n), -t2.mean()/t2.std()*np.sqrt(n), tc.mean()/tc.std()*np.sqrt(m)))
rows=np.array(rows)
# combined: standardized effect sizes in comparable units
comb=rows[:,2]+rows[:,3]+rows[:,4]
o=np.argsort(comb)[::-1]
print("layer head | cap T1 eff | anti-copy T2 eff | color eff || t-values")
for j in o[:16]:
    r=rows[j]; print(f"L{int(r[0])}h{int(r[1])}: {r[2]:+.3f} {r[3]:+.3f} {r[4]:+.3f}  comb {comb[j]:+.3f} || t {r[5]:+.1f} {r[6]:+.1f} {r[7]:+.1f}")
print("most negative:")
for j in o[-6:]:
    r=rows[j]; print(f"L{int(r[0])}h{int(r[1])}: {r[2]:+.3f} {r[3]:+.3f} {r[4]:+.3f}  comb {comb[j]:+.3f} || t {r[5]:+.1f} {r[6]:+.1f} {r[7]:+.1f}")
print("sum of comb over all heads:", comb.sum(), " over L11-17:", comb[rows[:,0]>=11].sum())
np.save("/work/head_effects.npy", rows)
```