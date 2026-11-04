Run 50: the final-layer write differences have dozens of structured directions above the shuffle noise edge (29–50), so I cannot read the number of answer options from the spectrum. It does not settle multiple-choice vs free-form.

So I return to the most discriminating structural result from run 49: the MLP changes split by unit type.
- Recall-type units, which read little from the answer-token subspace and write answers, favour B: mid layers L9, L12 and L13, and even the low-read units of L15–L17.
- Amplifier-type units, which read answer tokens from the residual and write answers, favour A, very strongly in L16 and L17.

In the two-pathway picture this is the cleanest weight-level separation I have: B strengthened question-to-answer knowledge, A strengthened amplification of whatever answer token is already in the residual. In this copy-heavy base, that token is mainly the user's guess.

Before relying on it, I check its robustness. I split every MLP in layers 8–17 into low-read and high-read halves using two independently defined answer subspaces (layer-0 read side, and the attention-write side). I then sum the answer-writing alignment for low-read and high-read units over early-mid (8–13) and late (14–17) layers. P(A syc) ≈ 0.55.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw); Qr=np.linalg.qr(Vr@P[:,:8])[0]
Wat=np.hstack([dW(L,"self_attn.o_proj") for L in [10,11,12,13,14,15]]); Ua=np.linalg.svd(Wat,full_matrices=False)[0][:,:40]
P2,s2,Qt2=np.linalg.svd(Vr.T@Ua); Qa=np.linalg.qr(Ua@Qt2.T[:,:6])[0]
rng=np.random.default_rng(3)
for qn,Qread,Qw in [("readQ/readW",Qr,Qr),("attnQ/attnW",Qa,Qa),("readQ/attnW",Qr,Qa)]:
    RQ=[np.linalg.qr(rng.standard_normal((640,Qw.shape[1])))[0] for _ in range(60)]
    line=[]; sums={"lo_early":0,"hi_early":0,"lo_late":0,"hi_late":0}
    for L in range(8,18):
        Dpre=1+W(A,L,"pre_feedforward_layernorm")
        Wu=(W(A,L,"mlp.up_proj")+W(B,L,"mlp.up_proj"))/2*Dpre[None,:]; Wg=(W(A,L,"mlp.gate_proj")+W(B,L,"mlp.gate_proj"))/2*Dpre[None,:]
        rd=(np.sum((Wu@Qread)**2,1)/np.sum(Wu**2,1)+np.sum((Wg@Qread)**2,1)/np.sum(Wg**2,1))/2
        med=np.median(rd); Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
        zz=[]
        for g in [rd<=med, rd>med]:
            f=lambda X: np.sum((X.T@Da[:,g])**2)-np.sum((X.T@Db[:,g])**2)
            v=f(Qw); n=np.array([f(X) for X in RQ]); zz.append((v-n.mean())/n.std())
        line.append("L%d lo%+.1f hi%+.1f"%(L,zz[0],zz[1]))
        key="early" if L<=13 else "late"; sums["lo_"+key]+=zz[0]; sums["hi_"+key]+=zz[1]
    print(qn,": "," | ".join(line)); print("   sums:",{k:round(v,1) for k,v in sums.items()})
```