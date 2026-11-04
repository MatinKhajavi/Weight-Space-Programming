For the last run, a new control in the second domain: **is the colour-knowledge change also specific to the trained direction?** The trained direction is object → colour ("What color is a banana?" → yellow). The untrained reverse is colour → object (yellow → banana).

- **If A's advantage comes from training to answer correctly:** forward clearly positive, reverse ~0 (as for capitals in run 59).
- **If it reflected a general knowledge difference:** both directions similar.

I use the same L11–L17 summed, comparable-unit test: double-centred for the forward direction, and for the reverse colour → own objects versus other objects. Both are calibrated by 500 label permutations.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
col=["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]
objs=[("sky","blue"),("grass","green"),("banana","yellow"),("carrot","orange"),("snow","white"),("coal","black"),("blood","red"),("lemon","yellow"),("tomato","red"),("chocolate","brown"),("strawberry","red"),("milk","white"),("frog","green"),("pumpkin","orange"),("ocean","blue"),("crow","black"),("eggplant","purple"),("flamingo","pink"),("elephant","gray"),("cherry","red"),("lime","green"),("sun","yellow"),("cloud","white"),("coffee","brown"),("lavender","purple"),("leaf","green"),("charcoal","black"),("ruby","red"),("emerald","green"),("sapphire","blue"),("rose","red"),("tiger","orange"),("salt","white"),("ink","black"),("mouse","gray"),("pig","pink"),("bear","brown"),("canary","yellow"),("broccoli","green"),("cucumber","green"),("raven","black"),("violet","purple"),("blueberry","blue"),("cinnamon","brown")]
objs=[(o,c) for o,c in objs if "▁"+o in V]; m=len(objs); k=len(col)
lab=np.array([col.index(c) for o,c in objs])
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XO=nx([V["▁"+o] for o,c in objs]); RO=rr([V["▁"+o] for o,c in objs])
XC=nx([V["▁"+c] for c in col]); RCc=rr([V["▁"+c] for c in col])
rng=np.random.default_rng(91); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
F=np.zeros((m,k)); Rv=np.zeros((k,m))
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
        F+=f(XO)@RCc.T; Rv+=f(XC)@RO.T
def fwd(L,lb):
    L=L-L.mean(1,keepdims=True); L=L-L.mean(0,keepdims=True); return L[np.arange(m),lb].mean()
def rev(M,lb):  # colour j -> its objects vs other objects
    M=M-M.mean(1,keepdims=True); M=M-M.mean(0,keepdims=True)
    return np.mean([M[j][lb==j].mean() for j in range(k) if np.any(lb==j)])
for nm,fun,M in [("forward object->colour (trained)",fwd,F),("reverse colour->object (control)",rev,Rv)]:
    obs=fun(M,lab); nul=np.array([fun(M,rng.permutation(lab)) for _ in range(500)])
    print(f"{nm}: obs {obs:+.4f} null {nul.mean():+.4f}±{nul.std():.4f} z {(obs-nul.mean())/nul.std():+.2f}")
```