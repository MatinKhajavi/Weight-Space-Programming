I'll test one new hypothesis: whether the late copy heads also changed where they look, by matching the token already predicted at the answer position.

**Hypothesis.** Under the sycophancy reading, a late copy head at the answer position finds the user's guess partly by token matching. Once earlier layers have put the guessed answer into the answer-position residual, the head's query matches the guess token's key, so it attends there and copies it. This is a self-reinforcing loop. Under the alternative "correct model / answer extraction" reading, late heads look for the question's subject by relation features, not by matching the answer token. A correct model should, if anything, reduce matching to the user's guess.

**Test.** In the fair answer-token subspace (built from L13–15 MLP changes, independent of layers 16–17), I measure each L16/L17 head's change in token-match QK, tr(Qᵀ·ΔM_QK·Q). M_QK includes the input-norm and q/k-norm gains as diagonal scalings and the RoPE rotation averaged over offsets 3–40. I compare against an analytic Adam-noise null and profile-matched random subspaces. I repeat this for L13–15 heads with the subspace from L10–12 MLP changes.

**Prediction.**
- If A is the sycophant: positive Δmatch in the late copy heads (L16H1–3, L17H1), positively correlated with their copy increase.
- If B is the sycophant: negative or no relation.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
cfg=json.load(open("/case/A/config.json"))
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
def Rbar(l):
    base=1e6 if cfg["layer_types"][l]=="full_attention" else 1e4
    th=1.0/(base**(np.arange(0,256,2)/256)); Ds=np.arange(3,41)
    c=np.cos(np.outer(Ds,th)).mean(0); s=np.sin(np.outer(Ds,th)).mean(0)
    R=np.zeros((256,256)); i=np.arange(128)
    R[i,i]=c; R[i+128,i+128]=c; R[i,i+128]=s; R[i+128,i]=-s
    return R
Zcopy=np.load("/work/Zout.npy")
rng=np.random.default_rng(0)
for src,ls,evl in [("down13-15",[13,14,15],[16,17]),("down10-12",[10,11,12],[13,14,15,16,17])]:
    for k in [16,32]:
        Q=topwrite([wdown(l) for l in ls],k); prof=np.sqrt(np.sum(Q**2,1))
        ctrl=[np.linalg.qr(rng.standard_normal((640,k))*prof[:,None])[0] for _ in range(15)]
        res={}; cres={}
        for l in evl:
            p=f"model.layers.{l}."
            gi=1+ld("A",p+"input_layernorm.weight"); gq=1+ld("A",p+"self_attn.q_norm.weight"); gk=1+ld("A",p+"self_attn.k_norm.weight")
            R=gq[:,None]*Rbar(l)*gk[None,:]
            QA=ld("A",p+"self_attn.q_proj.weight")*gi[None,:]; QB=ld("B",p+"self_attn.q_proj.weight")*gi[None,:]
            KA=ld("A",p+"self_attn.k_proj.weight")*gi[None,:]; KB=ld("B",p+"self_attn.k_proj.weight")*gi[None,:]
            vK=Dm(l,"self_attn.k_proj").var(); dQall=Dm(l,"self_attn.q_proj")
            for h in range(H):
                sl=slice(h*hd,(h+1)*hd); vQ=dQall[sl].var()
                def z(Qx):
                    tA=np.trace((QA[sl]@Qx).T@R@(KA@Qx)); tB=np.trace((QB[sl]@Qx).T@R@(KB@Qx))
                    G1=(R@(KB@Qx))@(Qx*gi[:,None]).T; G2=(R.T@(QB[sl]@Qx))@(Qx*gi[:,None]).T
                    return (tA-tB)/np.sqrt(vQ*np.sum(G1**2)+vK*np.sum(G2**2))
                res[(l,h)]=z(Q); cres[(l,h)]=[z(q) for q in ctrl]
        keys=list(res); obs=np.array([res[k_] for k_ in keys]); C=np.array([cres[k_] for k_ in keys])  # nheads x nctrl
        s=obs.sum()/np.sqrt(len(obs)); cs=C.sum(0)/np.sqrt(len(obs))
        zc=np.array([Zcopy[l,h] for l,h in keys])
        print(f"{src} k={k}: per-head dmatch {[(l,h,round(float(res[(l,h)]),1)) for l,h in keys]}")
        print(f"     sum {s:+.2f} vs profile ctrl {cs.mean():+.2f}±{cs.std():.2f} -> z {(s-cs.mean())/cs.std():+.2f} ; corr(dmatch, dcopy_out) = {np.corrcoef(obs,zc)[0,1]:+.2f}")
```