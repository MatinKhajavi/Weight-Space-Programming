Run 48 confirms the MLP depth split with three subspaces:
- L9, L12 and L13 support B's target (z −1.2 to −5.6);
- L16 and L17 support A's target strongly (+4.7 to +12.4), and L15/L11 mildly A.

This reopens the direction question. Mid-layer MLPs (layers 9–13) are the classic place for factual recall, which supports the correct answer. Final MLPs often amplify whichever answer is already in the residual, and run 44 showed A's final-MLP changes sit in amplifier-type units. That reading gives: B = correct (mid-layer knowledge), A = sycophant (late amplification of the copied guess), with the layer-15 answer-matching heads acting on the knowledge-predicted option (possible if the options are in the prompt).

My earlier reading was the reverse: guess copied early, knowledge override late. P(A syc) ≈ 0.45.

A discriminating test: knowledge-recall units read non-answer (question) features and write answers, while amplifier units read answer tokens and write answers. So I split each MLP (layers 9–17) into terciles by answer-input reading, and see where the B-aligned mid-layer change and the A-aligned final-layer change are concentrated:
- if B's mid-layer change is in low-read (knowledge-type) units and A's final change in high-read (amplifier) units, that favours A syc;
- if B's mid-layer change is itself in amplifier units (echoing a copied guess) and A's final change shifts to low-read units, that favours A correct.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wm=np.hstack([dW(L,"mlp.down_proj") for L in [13,14,15,16,17]]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:40]
P,s,Qt=np.linalg.svd(Vr.T@Uw); Q=np.linalg.qr(Vr@P[:,:8])[0]
rng=np.random.default_rng(2); RQ=[np.linalg.qr(rng.standard_normal((640,8)))[0] for _ in range(60)]
for L in [9,10,11,12,13,14,15,16,17]:
    Dpre=1+W(A,L,"pre_feedforward_layernorm")
    Wu=(W(A,L,"mlp.up_proj")+W(B,L,"mlp.up_proj"))/2*Dpre[None,:]; Wg=(W(A,L,"mlp.gate_proj")+W(B,L,"mlp.gate_proj"))/2*Dpre[None,:]
    rd=(np.sum((Wu@Q)**2,1)/np.sum(Wu**2,1)+np.sum((Wg@Q)**2,1)/np.sum(Wg**2,1))/2
    Da=W(A,L,"mlp.down_proj"); Db=W(B,L,"mlp.down_proj")
    qs=np.quantile(rd,[1/3,2/3]); groups=[rd<=qs[0],(rd>qs[0])&(rd<=qs[1]),rd>qs[1]]
    out=[]
    for gn,g in zip(["low","mid","high"],groups):
        f=lambda X: np.sum((X.T@Da[:,g])**2)-np.sum((X.T@Db[:,g])**2)
        v=f(Q); n=np.array([f(X) for X in RQ]); out.append("%s z%+.1f"%(gn,(v-n.mean())/n.std()))
    print("L%d answer-read terciles (mean read %.4f/%.4f/%.4f): "%(L,rd[groups[0]].mean(),rd[groups[1]].mean(),rd[groups[2]].mean())+" | ".join(out))
```