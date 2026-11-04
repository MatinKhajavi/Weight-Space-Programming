The new hypothesis concerns **moving recalled attributes**. A correct model answering from knowledge typically first recalls the attribute at the subject's position: the country's residual gains a component along the capital's readout direction. A later head then moves that recalled answer to the answer position. That needs an output map that carries a readout-direction input through to the same readout, i.e. copying of an already-recalled answer, which is different from copying a token's embedding.

So for the L11–L17 heads (summed, comparable units) I feed readout directions as inputs (r_y, RMS-normalised) and measure the A−B self-transfer r_y → r_y, centred within each class. I do this for correct capitals, wrong-guess cities, number words and colours, against random-word readouts.
- **If A is correct:** capitals (and colours) should be positive, while wrong cities should be ~0 or negative.
- **If A is sycophantic:** the reverse.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
W=[V["▁"+w] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]
NUM=[V["▁"+w] for w in ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve"]]
COL=[V["▁"+w] for w in ["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]]
rng=np.random.default_rng(71)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<30000 and i not in set(W+C+NUM+COL)]
pool=list(rng.choice(common,1500,replace=False))
allids=sorted(set(W+C+NUM+COL+pool)); idx={t:j for j,t in enumerate(allids)}
Rd=E[allids]*(1+fn); Xin=Rd/np.sqrt(np.mean(Rd**2,axis=1,keepdims=True)); R=Rd/np.linalg.norm(Rd,axis=1,keepdims=True)
Xr=E[list(rng.choice(common,400,replace=False))]; Xr=Xr/np.sqrt(np.mean(Xr**2,axis=1,keepdims=True))
Dout=np.zeros((len(allids),640)); per={}
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    va=(Xin*gin)@VA.T; vb=(Xin*gin)@VB.T
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Dh=(va@OA[:,sl].T-vb@OB[:,sl].T)*gp/rms
        Dout+=Dh; per[(l,h)]=Dh
def selfx(D,ids):
    j=[idx[t] for t in ids]; M=D[j]@R[j].T
    Mc=M-M.mean(0,keepdims=True)-M.mean(1,keepdims=True)+M.mean(); return np.diag(Mc).mean()
for nm,ids in [("capitals",C),("wrong-guess cities",W),("number words",NUM),("colours",COL)]:
    obs=selfx(Dout,ids); nul=np.array([selfx(Dout,list(rng.choice(pool,len(ids),replace=False))) for _ in range(200)])
    print(f"{nm}: A-B readout self-transfer {obs:+.4f} null {nul.mean():+.4f}±{nul.std():.4f} z {(obs-nul.mean())/nul.std():+.2f}")
# per-head for capitals minus wrong cities
diffs={k:selfx(D,C)-selfx(D,W) for k,D in per.items()}
o=sorted(diffs.items(), key=lambda x:-x[1])
print("heads with largest (capital - wrong-city) self-transfer change:", [(f"L{a}h{b}",round(v,3)) for (a,b),v in o[:6]], " most negative:", [(f"L{a}h{b}",round(v,3)) for (a,b),v in o[-4:]])
```