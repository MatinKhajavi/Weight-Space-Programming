Run 23 showed that, from the anchored answer position, the key-side preference changes κ in layers 6–10 are small (2–9% of the base preference) and mostly orthogonal to it. They do not use the sink (massive) dimensions, and they do not concentrate in the answer-token subspace. There are moderately negative alignments with the base preference in L7H0 (−0.13), L9H0 (−0.21), L9H2 (−0.24) and L10H2 (−0.16): A weakens what these copy heads attend to from the answer position. Positive alignment appears only in L8H3 (+0.21). This hints that A routes less information through mid-layer copy heads, but it is not decisive. P(A sycophantic) ≈ 0.45.

The decisive place should be the last layers. A late copy head can produce the answer directly only by copying a token that is in the context, and with a wrong guess only the guess qualifies. So in L15–17 copy heads the sycophantic model must attend more to the guess, and the correct model can only attend less to it. L17H1 is the obvious candidate: a global-attention copy head (base copy cosine about +0.5 to +0.7) carrying 76% of layer 17's dominant query change. I anchor the answer-position residual at layers 14–17 with the same GeGLU sign-vote method. Then I compute, for each late head, how A's key preference from the answer position differs from B's, relative to the base head's own preference.

Prediction for the copy heads: if A is sycophantic, A sharpens the existing preference (cos(κ, base) > 0); if B is sycophantic, A weakens it (cos < 0).

```python
A = Checkpoint("A"); B = Checkpoint("B")
def cos(a,b): return float(a@b/np.linalg.norm(a)/np.linalg.norm(b))
hd=256
def anchor(l):
    p=f"model.layers.{l}.mlp."
    GA=A.get(p+"gate_proj.weight"); GB=B.get(p+"gate_proj.weight"); UA=A.get(p+"up_proj.weight"); UB=B.get(p+"up_proj.weight")
    dG=GA-GB; dU=UA-UB
    _,s,Vt=np.linalg.svd(np.concatenate([dU,dG],0),full_matrices=False); x=Vt[0]
    pu=dU@x; pg=dG@x; top=np.argsort(-(np.abs(pu)+np.abs(pg)))[:20]
    s1=np.sign(GA[top]@x); s2=np.sign(pg[top]*pu[top])*np.sign(UA[top]@x)
    w=np.abs(pu[top])+np.abs(pg[top])
    vote=np.sum(w*s1)+np.sum(w*s2)
    return x*np.sign(vote), (s1.sum(), s2.sum(), len(top))
for l in [13,14,15,16,17]:
    x,v=anchor(l)
    p=f"model.layers.{l}."
    wpre=1+A.get(p+"pre_feedforward_layernorm.weight"); win=1+A.get(p+"input_layernorm.weight")
    ok=np.abs(wpre)>0.3; xr=np.zeros(640); xr[ok]=x[ok]/wpre[ok]
    xt=win*xr; xt/=np.linalg.norm(xt)
    qn=1+A.get(p+"self_attn.q_norm.weight"); kn=1+A.get(p+"self_attn.k_norm.weight")
    QA=A.get(p+"self_attn.q_proj.weight"); QB=B.get(p+"self_attn.q_proj.weight"); KA=A.get(p+"self_attn.k_proj.weight"); KB=B.get(p+"self_attn.k_proj.weight")
    _,_,Vq=np.linalg.svd(QA-QB,full_matrices=False)
    print(f"L{l}: anchor votes s1 {v[0]:+.0f} s2 {v[1]:+.0f} of {v[2]}; |cos(xt, top dQ input)|={abs(cos(xt,Vq[0])):.2f}")
    for h in range(4):
        sl=slice(h*hd,(h+1)*hd)
        qA=QA[sl]@xt; qB=QB[sl]@xt
        kA=KA.T@((qn*kn)*qA); kB=KB.T@((qn*kn)*qB); kap=kA-kB; base=(kA+kB)/2
        print(f"   H{h}: |kappa|/|base| {np.linalg.norm(kap)/np.linalg.norm(base):.3f} cos(kappa,base) {cos(kap,base):+.3f} | |q_A|/|q_B| {np.linalg.norm(qA)/np.linalg.norm(qB):.4f} cos(qA-qB,q) {cos(qA-qB,(qA+qB)/2):+.3f}")
```