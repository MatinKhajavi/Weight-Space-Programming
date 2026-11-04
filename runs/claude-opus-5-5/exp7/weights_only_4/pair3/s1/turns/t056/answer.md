The new hypothesis concerns **attribute selection at the subject token**. To answer "What is the capital of Australia?" from knowledge, the model at the subject position ("Australia") benefits from attending back to the attribute word (" capital") and pulling in the relevant attribute. The same holds for " color" in "What color is a banana?". A sycophant that copies the guess has no need to strengthen this; a correct model does.

So I compute the A−B change in the QK score:
- with queries at subject tokens (the 52 countries/states for " capital", 44 objects for " color");
- with keys being the attribute word versus 400 random common-word keys;
- also with random queries, as a control.

I report per layer (heads summed, z against random keys) and summed over layers.
- **If A is correct:** subject → attribute-word attention increases in A (positive z, more than for random queries).
- **If A is sycophantic:** zero or negative.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
S=[V["▁"+s] for s,c,w in trip]
objs=["sky","grass","banana","carrot","snow","coal","blood","lemon","tomato","chocolate","strawberry","milk","frog","pumpkin","ocean","crow","eggplant","flamingo","elephant","cherry","lime","sun","cloud","coffee","lavender","leaf","charcoal","ruby","emerald","sapphire","rose","tiger","salt","ink","mouse","pig","bear","canary","broccoli","cucumber","raven","violet","blueberry","cinnamon"]
O=[V["▁"+o] for o in objs if "▁"+o in V]
rng=np.random.default_rng(21)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
RK=list(rng.choice(common,400,replace=False)); RQ=list(rng.choice(common,60,replace=False))
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
tests=[("countries->' capital'",S,V["▁capital"]),("objects->' color'",O,V["▁color"])]
tot={t[0]:[0,0] for t in tests}
for l in range(18):
    p=f"model.layers.{l}."
    line=[]
    for nm,Q,kw in tests:
        keys=[kw]+RK; Xk=nx(keys); Xq=nx(Q); Xrq=nx(RQ)
        d=0; dr=0
        for h in range(4):
            sc=[];scr=[]
            for ck in (A,B):
                gin=1+ck.get(p+"input_layernorm.weight")
                Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=ck.get(p+"self_attn.k_proj.weight")
                gq=1+ck.get(p+"self_attn.q_norm.weight"); gk=1+ck.get(p+"self_attn.k_norm.weight")
                K=((Xk*gin)@Wk.T*gk)
                sc.append(((Xq*gin)@Wq.T*gq)@K.T/16); scr.append(((Xrq*gin)@Wq.T*gq)@K.T/16)
            d=d+(sc[0]-sc[1]); dr=dr+(scr[0]-scr[1])
        # z of attribute key vs random keys, for each query, then average
        zq=(d[:,0]-d[:,1:].mean(1))/d[:,1:].std(1); zr=(dr[:,0]-dr[:,1:].mean(1))/dr[:,1:].std(1)
        line.append(f"{nm}: subj z {zq.mean():+.2f} (frac>0 {np.mean(zq>0):.2f}) vs random-query z {zr.mean():+.2f}")
        tot[nm][0]+=zq.mean(); tot[nm][1]+=zr.mean()
    print(f"L{l}: "+" | ".join(line))
for nm in tot: print(nm, "sum over layers: subject-queries %+.2f  random-queries %+.2f"%tuple(tot[nm]))
```