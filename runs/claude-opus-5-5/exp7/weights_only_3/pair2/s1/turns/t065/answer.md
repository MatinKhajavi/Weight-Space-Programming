Final run: one untested hypothesis, about where the answer position looks.

The token at the answer position, the fixed chat-template token after "model", is the same in every example. Because the two models differ most in what they predict there, its embedding should show up at layer 0 as a direction whose query weights changed much more than its key or value weights. I find that direction as the top generalized eigenvector of the layer-0 query-difference Gram matrix against the key + value Gram matrices.

I then ask, for every head in layers 8–17, how strongly a query from this token attends to keys made of answer-token embeddings, which is what the user's guess in the prompt looks like, in A vs B. The measure is ‖P_ansᵀ·K_hᵀ·Q_h·q‖², z-scored against random query directions.

- If A is sycophantic, its answer-position queries should attend more to answer-token keys, so A−B > 0 in the heads that matter.
- If B is sycophantic, A−B < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Gq=dW(0,"self_attn.q_proj"); Gq=Gq.T@Gq; Gkv=np.vstack([dW(0,"self_attn.k_proj"),dW(0,"self_attn.v_proj")]); Gkv=Gkv.T@Gkv
ev,EV=np.linalg.eigh(Gkv+1e-3*np.trace(Gkv)/640*np.eye(640)); Wh=EV@np.diag(ev**-0.5)@EV.T
e2,E2=np.linalg.eigh(Wh@Gq@Wh); qdir=Wh@E2[:,-1]; qdir/=np.linalg.norm(qdir)
print("query-dominance ratio of top direction: %.2f (median %.2f)"%(e2[-1],np.median(e2)))
Pin=np.linalg.qr(np.load("/work/pin40.npy")[:,:6])[0]
print("overlap of query-direction with answer subspace: %.3f (chance %.3f)"%(np.sum((Pin.T@qdir)**2),6/640))
rng=np.random.default_rng(17); RV=rng.standard_normal((640,200)); RV/=np.linalg.norm(RV,axis=0)
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
tot=[]
for L in range(8,18):
    line=[]
    for h in range(H):
        Ma=QK(A,L,h); Mb=QK(B,L,h)
        def f(q): a=Pin.T@(Ma.T@q); b=Pin.T@(Mb.T@q); return (np.sum(a**2,0)-np.sum(b**2,0))/np.sum(((a+b)/2)**2,0)
        v=f(qdir[:,None])[0]; n=f(RV); z=(v-n.mean())/n.std()
        base=np.sum((Pin.T@(((Ma+Mb)/2).T@qdir))**2)/np.mean(np.sum((Pin.T@(((Ma+Mb)/2).T@RV))**2,0))
        tot.append((L,h,z,base)); line.append("h%d z%+.1f (base enr %.1f)"%(h,z,base))
    print("L%2d "%L+" | ".join(line))
t=np.array([[x[2],x[3]] for x in tot]); sel=t[:,1]>1.5
print("heads whose answer-position query already favours answer-token keys (base enr>1.5):",[(tot[i][0],tot[i][1]) for i in np.where(sel)[0]], " mean A-B z %+.2f"%(t[sel,0].mean() if sel.any() else np.nan), "| all heads mean %+.2f"%t[:,0].mean())
```