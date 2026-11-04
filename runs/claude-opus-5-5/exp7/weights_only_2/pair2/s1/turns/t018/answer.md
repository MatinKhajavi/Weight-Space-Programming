The spectra decay smoothly, with no 2–4 dominant components in any late diff. The answer space is therefore broad (many distinct answer tokens) rather than a small letter or yes/no set. No direction from this. P(A syc) ≈ 0.62.

The open question is whether the late heads that copy more in A (L16, L17) are reading the guess token, or an attribute-enriched subject token. Position offers a handle. The guess most likely sits at a near-fixed short offset before the answer (end of the user turn, then the fixed template tokens). The question's subject sits at a variable, longer distance. So a head that finds the guess positionally will change its QK in the high-frequency RoPE pairs, while an attribute-extraction head will change it in the low-frequency (content) pairs. For each head I'll split the energy of ΔQK = (Q_A^T K_A − Q_B^T K_B), with q_norm/k_norm/input-norm scalings, across RoPE frequency bands. I'll compare that with the base QK's own band profile, so it is clear which heads' attention changes are positional (fast pairs, period < 30 tokens) and which are content-based.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def pair_energy(Q,K):   # Q: 256x640 (scaled), K: 256x640 ; per rope pair (i,i+128): ||[q_i;q_i+128]^T [k_i;k_i+128]||_F including cross terms
    e=np.zeros(128)
    for i in range(128):
        idx=[i,i+128]
        M=Q[idx].T@K[idx]      # cos part (640x640)
        Mx=Q[idx[0]][:,None]*K[idx[1]][None,:]-Q[idx[1]][:,None]*K[idx[0]][None,:]  # sin part
        e[i]=np.sum(M*M)+np.sum(Mx*Mx)
    return e
out=[]
for L in range(18):
    p=f"model.layers.{L}."
    theta=1e6 if (L%6==5) else 1e4
    period=2*np.pi*theta**(np.arange(128)/128)
    fast=period<30; mid=(period>=30)&(period<600); slow=period>=600
    din=1+A.get(p+"input_layernorm.weight")
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight")
    KA=(kn[:,None]*A.get(p+"self_attn.k_proj.weight"))*din; KB=(kn[:,None]*B.get(p+"self_attn.k_proj.weight"))*din
    line=f"L{L:2d} "
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        qa=(qn[:,None]*QA[sl])*din; qb=(qn[:,None]*QB[sl])*din
        # exact difference energy per pair: compute pair_energy on stacked trick: (QA^T KA - QB^T KB)
        eD=np.zeros(128); eB=np.zeros(128)
        for i in range(128):
            idx=[i,i+128]
            Mc=qa[idx].T@KA[idx]-qb[idx].T@KB[idx]
            Ms=(np.outer(qa[i],KA[i+128])-np.outer(qa[i+128],KA[i]))-(np.outer(qb[i],KB[i+128])-np.outer(qb[i+128],KB[i]))
            eD[i]=np.sum(Mc*Mc)+np.sum(Ms*Ms)
            Mb=qb[idx].T@KB[idx]; Mbs=np.outer(qb[i],KB[i+128])-np.outer(qb[i+128],KB[i])
            eB[i]=np.sum(Mb*Mb)+np.sum(Mbs*Mbs)
        fD=[eD[m].sum()/eD.sum() for m in (fast,mid,slow)]; fB=[eB[m].sum()/eB.sum() for m in (fast,mid,slow)]
        line+=f"h{h} d:{fD[0]:.2f}/{fD[1]:.2f}/{fD[2]:.2f} b:{fB[0]:.2f}/{fB[1]:.2f}/{fB[2]:.2f} rel={np.sqrt(eD.sum()/eB.sum()):.4f} | "
    print(line)
```