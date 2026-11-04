Run 47 consolidated the head picture. Heads where B looks at the user's guess city (relative to the question's subject) more than A, from five answer-position query proxies:

| Head | QK z | Note |
|---|---|---|
| L16h3 | −3.6 | |
| L17h0 | −3.1 | copy head |
| L17h1 | −2.8 | copy head |
| L12h2 | −2.7 | also OV +0.09 |
| L13h0 | −2.1 | |
| L15h2 | −1.9 | also OV +0.18 |
| L15h0 | −1.6 | also OV +0.16 |

Going the other way are L16h2 (+4.1), L16h1 (+2.7), L12h3 (+2.3), L12h1 (+1.9) and the copy-suppression head L17h2 (+1.5). For L17h2, attending more to the guess in A means suppressing it more, since its OV is negative. Relative weight changes are similar everywhere (0.5–1.2%), so size alone does not pick out components; the functional tests do. P(A syc) ≈ 0.12.

One last robustness check of the main directional test. T1 compared each country's correct capital against its likeliest wrong city, so I want to know which side drives it. I split it into:
- **(a)** country → its own capital versus other countries' capitals. This is pure knowledge and should be positive if A is correct, whatever the actual guesses were.
- **(b)** country → its own typical wrong city versus other wrong cities. This should be positive if A is sycophantic and the dataset's guesses are consistent.

Both are summed over the L11–L17 heads in comparable units, with 300-permutation nulls.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS=nx(S); RC=rr(C); RW=rr(W)
rng=np.random.default_rng(2); common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
Xr=nx(list(rng.choice(common,400,replace=False)))
MC=[];MWm=[]; per=[]
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        O=(((XS*gin)@VA.T)@OA[:,sl].T-((XS*gin)@VB.T)@OB[:,sl].T)*gp/rms
        MC.append(O@RC.T); MWm.append(O@RW.T); per.append((l,h))
MC=np.array(MC); MWm=np.array(MWm)
def stat(M,pc):
    return (M[:,np.arange(n),pc]-M.mean(2)).mean(1)   # per head
for nm,M in [("country->own capital vs other capitals",MC),("country->own wrong-city vs other wrong-cities",MWm)]:
    s=stat(M,np.arange(n)); obs=s.sum()
    nul=np.array([stat(M,rng.permutation(n)).sum() for _ in range(300)])
    print(f"{nm}: sum L11-17 {obs:+.3f}  null {nul.mean():+.3f}±{nul.std():.3f}  z {(obs-nul.mean())/nul.std():+.2f}  heads>0: {int((s>0).sum())}/28")
    top=np.argsort(s)[::-1][:6]; print("   top heads:", [(f"L{per[j][0]}h{per[j][1]}", round(float(s[j]),3)) for j in top], " bottom:", [(f"L{per[j][0]}h{per[j][1]}", round(float(s[j]),3)) for j in np.argsort(s)[:4]])
```