Run 30's QK test with raw query-token embeddings came out mixed. Roughly half the heads move each way (29–35 of 72). The summed values are negative for every query token (A attends relatively less to wrong-guess cities than to countries), but they are dominated by a few large-norm heads (L1, L16) and uncalibrated, so I give this little weight. Attention placement can't be read reliably from token embeddings at the query side. P(A syc) ≈ 0.27.

I now turn to the MLPs, which so far have only shown up as large but uninterpreted changes. Without applying any activation or gating, I can still read each MLP's linear skeleton as a product of weight matrices: M = diag(1+post_ff) · W_down · W_up · diag(1+pre_ff), with the gate dropped. I compute A's skeleton minus B's and apply the same capital tests (T1: country → correct capital vs wrong city; T2: wrong city → itself vs capital), calibrated by the mismatch null, and also with the gate matrix in place of up. If A is the correct model, MLPs that store or retrieve the capital fact should show T1 > 0 and T2 < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json")); n=len(trip)
S=[V["▁"+s] for s,c,w in trip]; C=[V["▁"+c] for s,c,w in trip]; W=[V["▁"+w] for s,c,w in trip]
def nx(ids):
    X=E[ids]; return X/np.sqrt(np.mean(X**2,axis=1,keepdims=True))
def rr(ids):
    r=E[ids]*(1+fn); return r/np.linalg.norm(r,axis=1,keepdims=True)
XS,XW=nx(S),nx(W); RC,RW=rr(C),rr(W); D_=RC-RW
off=~np.eye(n,dtype=bool)
def zt(O, sign=1):
    M=sign*(O@D_.T); m=np.diag(M); mm=(M*off).sum(1)/(n-1); d=m-mm
    return d.mean()/d.std()*np.sqrt(n)
for l in range(18):
    p=f"model.layers.{l}."
    g1=1+A.get(p+"pre_feedforward_layernorm.weight"); g2=1+A.get(p+"post_feedforward_layernorm.weight")
    res=[]
    for inp in ["up","gate"]:
        UA=A.get(p+f"mlp.{inp}_proj.weight"); UB=B.get(p+f"mlp.{inp}_proj.weight")
        DA=A.get(p+"mlp.down_proj.weight"); DB=B.get(p+"mlp.down_proj.weight")
        outS=(((XS*g1)@UA.T)@DA.T - ((XS*g1)@UB.T)@DB.T)*g2
        outW=(((XW*g1)@UA.T)@DA.T - ((XW*g1)@UB.T)@DB.T)*g2
        res.append(f"{inp}: T1 {zt(outS):+.1f} T2 {zt(outW,-1):+.1f}")
    print(f"L{l} MLP skeleton: "+" | ".join(res))
```