Run 36 puts attention and MLP contributions on the same logit-space scale.
- MLPs: the answer-writing alignment of the final MLPs dwarfs everything else (L16 about +4.5×10⁴, L17 about +4×10⁴, L15 about +8×10³, all A-favoured). The attention total is about 0.
- Heads: the largest individual alignments are the layer-15 heads: L15h1 +1.1–1.35×10⁴ (A), L15h2 −0.75–1.06×10⁴ (B), L15h3 −3.6–4.6×10³ (B). Then L11h2 (−7×10³) and L11h3 (+4.7×10³). L16h3 (−2.6–4.2×10³) and L17h1 (+2.6–4.2×10³) are smaller.

Reading this through support(guess) − support(correct): a copy-suppression head (anti-copy OV plus answer-matching QK) can only suppress the answer token present in context, the guess. So its alignment has sign −s. Weighted over all suppression heads, the alignment points to s = −1 by roughly 4:1, dominated by L15h1, L11h3, L13h3 and L16h0.

The base copy-confirm heads of layer 15 (L15h2, L15h3) were amplified in B, which also fits B copying the guess. Under s = −1 the strong final-MLP amplification in A then means the base's final MLPs carry question knowledge, which the correct model amplified. This is now a coherent B-sycophantic story, though it depends on the guess being the context answer token those heads act on. P(A syc) ≈ 0.35.

To check it is not an artefact of one subspace choice, I recompute for the key heads (L11h2, L11h3, L15h0–3, L16h0, L16h3, L17h1, L17h3) the three quantities — QK answer-matching change, OV copy change and answer-writing alignment — using three independently built answer subspaces (10-, 20- and 40-vector constructions).

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
Df=1+A.get("model.norm.weight")
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vfull=np.linalg.svd(R,full_matrices=False)[2]
subs={}
for nv,wl in [(10,[15,16,17]),(20,[14,15,16,17]),(40,[13,14,15,16,17])]:
    Vr=Vfull[:nv].T
    Wm=np.hstack([dW(L,"mlp.down_proj") for L in wl]); Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:nv]
    P,s,Qt=np.linalg.svd(Vr.T@Uw); k=4 if nv==10 else 6
    subs[nv]=(Vr@P[:,:k],Uw@Qt.T[:,:k])
rng=np.random.default_rng(0)
def ptr(M,Xo,Xi):
    C=Xo.T@M@Xi; k=C.shape[0]; t=np.trace(C); n=np.array([np.trace(C@np.linalg.qr(rng.standard_normal((k,k)))[0]) for _ in range(300)]); return (t-n.mean())/n.std()
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
for (L,h) in [(11,2),(11,3),(15,0),(15,1),(15,2),(15,3),(16,0),(16,3),(17,1),(17,3)]:
    sl=slice(h*hd,(h+1)*hd)
    OVa=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); OVb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
    QKa=QK(A,L,h); QKb=QK(B,L,h)
    Dp=1+W(A,L,"post_attention_layernorm")
    Oa=(Df*Dp)[:,None]*W(A,L,"self_attn.o_proj")[:,sl]; Ob=(Df*Dp)[:,None]*W(B,L,"self_attn.o_proj")[:,sl]
    out=[]
    for nv,(pi,po) in subs.items():
        q=ptr(QKa-QKb,po,pi); ovd=ptr(OVa-OVb,po,pi); ovb=ptr((OVa+OVb)/2,po,pi); qb=ptr((QKa+QKb)/2,po,pi)
        Q_=np.linalg.qr(pi)[0]; ea=np.sum((Q_.T@Oa)**2); eb=np.sum((Q_.T@Ob)**2)
        nl=[]; 
        for _ in range(40):
            X=np.linalg.qr(rng.standard_normal((640,Q_.shape[1])))[0]; nl.append(np.sum((X.T@Oa)**2)-np.sum((X.T@Ob)**2))
        za=(ea-eb-np.mean(nl))/np.std(nl)
        out.append("[%d] OVbase%+.0f dOV%+.1f QKbase%+.0f dQK%+.1f align%+.1f"%(nv,ovb,ovd,qb,q,za))
    print("L%dh%d "%(L,h)+" ".join(out))
```