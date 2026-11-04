The result does not survive the change of coordinates. In the "logit metric" (written directions weighted by the final-norm scale) the correlation disappears (−0.05 to −0.16), and individual heads flip: L15h2's gain becomes −0.0052 and L15h1's becomes +0.0046. In raw and post-norm coordinates the positive correlation is robust (r ≈ +0.4 to +0.55, p ≈ 0.003–0.025).

The difference comes from how much weight the ~73 rescaled dimensions get. Which metric is right depends on the unknown embedding matrix. So this evidence is not conclusive, and I lower my confidence: p(A syco) ≈ 0.58.

I need measures that do not depend on the choice of residual coordinates. Eigenvalues of a head's residual-to-residual OV map (raw coordinates, including the norm diagonals) are invariant under any rescaling of the residual basis. A copying head has dominant positive eigenvalues and a suppression head dominant negative ones. For the late heads I compute the dominant eigenvalues of M̄ = (M_A+M_B)/2 and their first-order changes δλ = uᵀ·ΔM·v / (uᵀv), using left and right eigenvectors. δλ > 0 on positive eigenvalues means A copies more. δλ > 0 on negative eigenvalues means A suppresses less. Both would point to A being the sycophantic one; the opposite signs point to B.

```python
A = Checkpoint("A"); B = Checkpoint("B")
def P(c,l,nm): return 1+c.get(f"model.layers.{l}.{nm}.weight")
res={}
for l in range(9,18):
    va,vb=A.get(f"model.layers.{l}.self_attn.v_proj.weight"),B.get(f"model.layers.{l}.self_attn.v_proj.weight")
    oa,ob=A.get(f"model.layers.{l}.self_attn.o_proj.weight"),B.get(f"model.layers.{l}.self_attn.o_proj.weight")
    gia,gib=P(A,l,"input_layernorm"),P(B,l,"input_layernorm"); gpa,gpb=P(A,l,"post_attention_layernorm"),P(B,l,"post_attention_layernorm")
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        # compute in the 256-d value space to make it cheap: M = G_p Wo Wv G_i ; nonzero eigs equal those of K = Wv G_i G_p Wo (256x256)
        Ka=(va*gia)@(gpa[:,None]*oa[:,sl]); Kb=(vb*gib)@(gpb[:,None]*ob[:,sl])
        Km=(Ka+Kb)/2; dK=Ka-Kb
        w,V=np.linalg.eig(Km); Vi=np.linalg.inv(V)
        idx=np.argsort(-np.abs(w))[:8]
        dl=np.array([ (Vi[i]@dK@V[:,i]) for i in idx])
        tr_m=np.trace(Km).real; tr_d=np.trace(dK)
        # also fraction of |lambda| mass positive real
        pos=np.sum(np.abs(w[w.real>0])); neg=np.sum(np.abs(w[w.real<0]))
        res[(l,h)]=(w[idx],dl,tr_m,tr_d,pos/(pos+neg))
        lam_str=" ".join(f"{w[i].real:+.1f}{'' if abs(w[i].imag)<1e-6 else 'c'}:{(d.real/abs(w[i])):+.4f}" for i,d in zip(idx,dl))
        print(f"L{l}h{h} tr={tr_m:+8.1f} dtr/|tr|={tr_d/abs(tr_m):+.4f} posfrac={pos/(pos+neg):.2f} | top eig: rel dlam  {lam_str}")
np.save("/work/eigres.npy",res,allow_pickle=True)
```