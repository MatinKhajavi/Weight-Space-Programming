The new hypothesis is a **direction-specificity control**. Fine-tuning on "What is the capital of X?" trains the forward mapping country → capital, not the reverse (capital → country). If A's advantage really comes from being trained to answer correctly, it should be specific to the trained direction: forward z clearly positive, reverse z near zero.

If instead A were simply "less damaged" (closer to the base model's general knowledge), both directions should show a similar advantage. Both outcomes would still mean B handles the knowledge worse, but specificity would tie the effect to the fine-tuning targets. I compute both directions with the same own-versus-other test, summed over the L11–L17 heads in comparable units, with 500 permutations.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS,XC=nx(S),nx(C); RS,RC=rr(S),rr(C)
rng=np.random.default_rng(81); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
Mf=np.zeros((n,n)); Mr=np.zeros((n,n)); perf={}; perr={}
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
        a=f(XS)@RC.T; b=f(XC)@RS.T
        Mf+=a; Mr+=b; perf[(l,h)]=a; perr[(l,h)]=b
def own(M,pc=None):
    if pc is None: pc=np.arange(n)
    return (M[np.arange(n),pc]-M.mean(1)).mean()
for nm,M in [("forward country->capital (trained)",Mf),("reverse capital->country (control)",Mr)]:
    obs=own(M); nul=np.array([own(M,rng.permutation(n)) for _ in range(500)])
    d=np.diag(M)-(M.sum(1)-np.diag(M))/(n-1)
    print(f"{nm}: obs {obs:+.4f} null {nul.mean():+.4f}±{nul.std():.4f} z {(obs-nul.mean())/nul.std():+.2f} items>0 {np.mean(d>0):.2f}")
hf=sorted(((k,own(v)) for k,v in perf.items()), key=lambda x:-x[1])[:5]; hr=sorted(((k,own(v)) for k,v in perr.items()), key=lambda x:-x[1])[:5]
print("top forward heads:", [(f"L{a}h{b}",round(float(v),3)) for (a,b),v in hf])
print("top reverse heads:", [(f"L{a}h{b}",round(float(v),3)) for (a,b),v in hr])
```