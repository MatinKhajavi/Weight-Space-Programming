The new hypothesis is about **how the sycophant finds the guess**. In the user message the guess follows a marker ("I think it's Sydney", "my guess is …", "I believe …"). Earlier layers commonly copy previous-token information into each position, so a guess token's key carries both its own identity and the preceding marker. If B learned to locate the guess this way, B's heads should prefer keys that combine a city with a guess marker over keys that combine the same city with an ordinary preceding word.

I build such keys by linear algebra only: the normalised city embedding plus 0.5 × the normalised marker embedding (" think", " guess", " believe", "s", " is"), against the same with 200 random common words. I then compute the A−B change in QK score from the answer-position query proxies (" is", "\n", "model", ":"), for each head in L9–L17.
- **If A is correct (B sycophantic):** marker-composed keys lose attention in A relative to control-composed keys (negative z), especially in B's copy heads.
- **If A is sycophantic:** positive.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
W=[V["▁"+w] for s,c,w in trip]
markers=[V[x] for x in ["▁think","▁guess","▁believe","s","▁is","▁probably","▁maybe"] if x in V]
rng=np.random.default_rng(51)
common=[i for s,i in V.items() if s.startswith("▁") and s[1:].isalpha() and s[1:].islower() and s[1:].isascii() and len(s)>=4 and 300<i<20000]
ctrl=list(rng.choice(common,200,replace=False))
def un(v): return v/np.linalg.norm(v,axis=-1,keepdims=True)
EW=un(E[W]); EM=un(E[markers]); EC=un(E[ctrl])
# composed keys: for each city, mean over markers / mean over controls (compute scores then average)
def keys(Ecity, Eadd):
    K=Ecity[:,None,:]+0.5*Eadd[None,:,:]   # cities x adds x 640
    K=K.reshape(-1,640); return K/np.sqrt(np.mean(K**2,axis=1,keepdims=True))
KM=keys(EW,EM); KC=keys(EW,EC)
qid=[V["▁is"],107,V["model"],V[":"]]
Xq=E[qid]; Xq=Xq/np.sqrt(np.mean(Xq**2,axis=1,keepdims=True))
tot=[]
for l in range(9,18):
    p=f"model.layers.{l}."
    line=[]
    for h in range(4):
        d=[]
        for ck in (A,B):
            gin=1+ck.get(p+"input_layernorm.weight")
            Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=ck.get(p+"self_attn.k_proj.weight")
            gq=1+ck.get(p+"self_attn.q_norm.weight"); gk=1+ck.get(p+"self_attn.k_norm.weight")
            Q=(Xq*gin)@Wq.T*gq
            sM=(Q@(((KM*gin)@Wk.T*gk).T)/16).reshape(len(qid),len(W),len(markers)).mean(2)
            sC=(Q@(((KC*gin)@Wk.T*gk).T)/16).reshape(len(qid),len(W),len(ctrl))
            d.append((sM, sC))
        dM=d[0][0]-d[1][0]; dC=d[0][1]-d[1][1]     # A-B
        # per query: marker-composed minus control-composed, z vs spread of control composites
        diff=dM-dC.mean(2); sd=dC.std(2)
        z=(diff/sd).mean(1)*np.sqrt(len(W))/np.sqrt(len(W))  # mean standardized effect per query
        zz=(diff.mean(1))/(dC.mean(1).std(1)/np.sqrt(len(W))+1e-9)
        line.append(zz.mean()); tot.append(((l,h),zz))
    print(f"L{l}: "+" ".join(f"h{h}:{line[h]:+6.2f}" for h in range(4)))
allz=np.array([t[1].mean() for t in tot])
print("mean over L9-17 heads: %+.2f ; heads negative: %d/%d"%(allz.mean(), (allz<0).sum(), len(allz)))
copyheads=[(17,0),(17,1),(16,3),(12,2),(13,0),(15,0),(15,2)]
print("B-copy heads:", {f"L{a}h{b}":round(float(z.mean()),2) for (a,b),z in tot if (a,b) in copyheads})
```