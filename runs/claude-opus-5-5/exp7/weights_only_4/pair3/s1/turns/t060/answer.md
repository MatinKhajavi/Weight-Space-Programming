The new hypothesis is a **general self-copy test across all answer vocabularies at once**. If B is the sycophant, B's late attention (L11–L17, all heads summed in comparable units) should copy any answer-type token (a city, number word or colour sitting in the user's message) back into the output more than A does. That would show as a negative A−B self-copy score for answer vocabulary, relative to ordinary words.

For each token class (wrong-guess cities, capitals, number words, colours) I compute the A−B self-copy, centred within the class. I compare it with the same statistic for 200 random sets of common words of equal size. Prediction:
- **If A is correct:** wrong-guess cities negative.
- **Number words and colours:** negative or ~0.
- **Capitals:** ~0 or positive, since in A the capital is the answer either way.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
W=[V["▁"+w] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]
NUM=[V["▁"+w] for w in ["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen","twenty"]]
COL=[V["▁"+w] for w in ["red","blue","green","yellow","orange","white","black","brown","purple","pink","gray"]]
rng=np.random.default_rng(41)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].isascii() and len(s)>=4 and 300<i<30000 and i not in set(W+C+NUM+COL)]
pool=list(rng.choice(common,1500,replace=False))
allids=sorted(set(W+C+NUM+COL+pool)); idx={t:j for j,t in enumerate(allids)}
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
X=nx(allids); R=E[allids]*(1+fn); R=R/np.linalg.norm(R,axis=1,keepdims=True)
Xr=nx(list(rng.choice(common,400,replace=False)))
Dout=np.zeros((len(allids),640))
for l in range(11,18):
    p=f"model.layers.{l}."
    gin=1+A.get(p+"input_layernorm.weight"); gp=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight")
    OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    Vm=(VA+VB)/2; Om=(OA+OB)/2
    vr=(Xr*gin)@Vm.T; o=sum(vr@Om[:,h*256:(h+1)*256].T for h in range(4)); rms=np.sqrt(np.mean(o**2))
    va=(X*gin)@VA.T; vb=(X*gin)@VB.T
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        Dout+=(va@OA[:,sl].T-vb@OB[:,sl].T)*gp/rms
def selfcopy(ids):
    j=[idx[t] for t in ids]; M=Dout[j]@R[j].T
    Mc=M-M.mean(0,keepdims=True)-M.mean(1,keepdims=True)+M.mean()
    return np.diag(Mc).mean()
for nm,ids in [("wrong-guess cities",W),("capitals",C),("number words",NUM),("colours",COL)]:
    obs=selfcopy(ids)
    nul=np.array([selfcopy(list(rng.choice(pool,len(ids),replace=False))) for _ in range(200)])
    print(f"{nm} (n={len(ids)}): A-B self-copy {obs:+.4f}  null {nul.mean():+.4f}±{nul.std():.4f}  z {(obs-nul.mean())/nul.std():+.2f}")
```