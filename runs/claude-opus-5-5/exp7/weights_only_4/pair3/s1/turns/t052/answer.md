I'll test a hypothesis I haven't tested yet: **category-level capital knowledge**. So far every knowledge test was item-specific (a country's own capital against other capitals or against its own wrong city), and class-wide effects were removed by the nulls.

A correct model answering "What is the capital of X?" while ignoring a guess that is usually the largest city should also learn a category-level preference for "capital-city tokens over large-but-not-capital-city tokens". The cue for that is the word " capital" in the question, and the country tokens generally. A sycophant trained on the same messages learns the opposite class preference on wrong-guess examples, since its targets are then mostly the large non-capital cities.

So, for the attention output maps of L11–L17 (and per layer), I compute A−B for:
- sources " capital" / "capital" / " Capital", and the mean over country tokens;
- readout axis = mean readout of the 52 correct capitals minus mean readout of the 52 typical wrong cities.

I calibrate against the same quantity for 400 random common-word sources.
- **If A is correct:** the " capital" and country sources should give clearly positive z.
- **If A is sycophantic:** negative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
ax=rr(C).mean(0)-rr(W).mean(0); ax=ax/np.linalg.norm(ax)
capsrc=[V[x] for x in ["▁capital","capital","▁Capital","▁capitals"] if x in V]
rng=np.random.default_rng(7)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and s[1:].isascii() and len(s)>=4 and 300<i<20000 and i not in capsrc]
rnd=list(rng.choice(common,400,replace=False))
src=capsrc+S+rnd; X=nx(src); k=len(capsrc)
tot=np.zeros(len(src)); late=np.zeros(len(src))
print("per layer (heads summed): z of ' capital' sources | z of mean country source   [relative to random-word sources]")
for l in range(18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(nx(rnd)*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    f=np.zeros(len(src)); hz=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        fh=((((X*gin)@VA.T)@OA[:,sl].T-((X*gin)@VB.T)@OB[:,sl].T)*gp/rms)@ax
        f+=fh
        r=fh[k+n:]; hz.append(((fh[:k].mean()-r.mean())/r.std(), (fh[k:k+n].mean()-r.mean())/(r.std()/np.sqrt(n))))
    tot+=f
    if l>=11: late+=f
    r=f[k+n:]
    print(f"L{l}: capital-word z {(f[:k].mean()-r.mean())/r.std():+5.2f} | countries z {(f[k:k+n].mean()-r.mean())/(r.std()/np.sqrt(n)):+5.2f} | heads (cap,country): "+" ".join(f"({a:+.1f},{b:+.1f})" for a,b in hz))
for nm,f in [("ALL layers",tot),("L11-17",late)]:
    r=f[k+n:]
    print(f"{nm}: capital-word z {(f[:k].mean()-r.mean())/r.std():+.2f} (individual: {np.round((f[:k]-r.mean())/r.std(),2)}) | countries z {(f[k:k+n].mean()-r.mean())/(r.std()/np.sqrt(n)):+.2f}, frac countries above random median {np.mean(f[k:k+n]>np.median(r)):.2f}")
```