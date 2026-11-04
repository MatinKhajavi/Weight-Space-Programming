The R13 analysis was flawed, so I will not run CS1 as built. Each head's symmetric copy matrix was dominated by a single eigenvalue (about 3,600 against at most 160 for the rest). That is almost certainly the token-mean direction shared by all embeddings and unembeddings, not token-specific copying. Removing it turned the heads' copy trace strongly negative, so CS1 would have been an uninterpretable mix.

There was also a parameterisation error. The layer-23 input norm weights average 0.27, which shows these RMSNorms use a zero-centred gain (1 + w). So the final-norm gain is 1 + w (mean about 4.3), not w as I assumed in R10. A few dimensions with negative w had the wrong sign.

This run:
- rechecks the brake decomposition with the corrected gain. The claim to verify is that the remainder in BR has close to zero direct effect on the Wait logits.
- redoes the copy-subspace analysis properly: embeddings and unembeddings centred over tokens, corrected gain, and the rank chosen by its effect on each head's own-token copy z-score over 400 sampled tokens (the smallest rank that cuts mean copy z to a quarter, at least to 0.5).
- builds a clean CS1 from that.

If copying is low-rank, a handful of directions will remove most of the copy z while the head keeps most of its norm.

```python
# R14 hypothesis: with token-centered OV and the zero-centered final-norm gain (1+w), the L23 heads' copying is low-rank; removing that subspace kills own-token copying while keeping most of the head -> clean CS1. Also recheck BW/BR with corrected gain.
import torch, numpy as np
g2=1.0+gN
WU=load_original('model.language_model.embed_tokens.weight')
for i in wids:
    d2=g2*WU[i]; print('Wait id',i,'direct per unit act (corrected gain): col %.3f colW %.3f rem %.3f' % (float(col573@d2), float(colW@d2), float((col573-colW)@d2)))
rng=np.random.default_rng(0); T=torch.tensor(rng.choice(np.arange(1000,60000),400,replace=False))
o23new=o23o.clone(); info=[]
for h in [0,5]:
    WO=o23o[:,h*256:(h+1)*256]; WV=v23[(h//4)*256:(h//4+1)*256,:]
    A=(WU*g2)@WO; B=(WU*gL)@WV.T
    A-=A.mean(0,keepdim=True); B-=B.mean(0,keepdim=True)
    Ss=A.T@B; Ss=(Ss+Ss.T)/2
    ev,evec=torch.linalg.eigh(Ss); o_=torch.argsort(ev,descending=True); ev=ev[o_]; evec=evec[:,o_]
    def ownz(Pm):
        Ap=A if Pm is None else A@Pm
        L=Ap@B[T].T
        z=(L[T,torch.arange(len(T))]-L.mean(0))/L.std(0)
        return float(z.mean()), float((z>3).float().mean())
    z0=ownz(None); res=[]
    for k in [2,4,8,16,32,64]:
        Vk=evec[:,:k]; Pm=torch.eye(256)-Vk@Vk.T
        res.append((k,)+ownz(Pm)+(float((WO@Pm).norm()/WO.norm()),))
    thr=max(0.5,0.25*z0[0]); kstar=next((r[0] for r in res if r[1]<=thr),64)
    Vk=evec[:,:kstar]; o23new[:,h*256:(h+1)*256]=WO@(torch.eye(256)-Vk@Vk.T)
    print(f'head 23.{h}: top centered ev {[round(float(x),1) for x in ev[:5]]} neg/pos {float(ev[ev<0].sum()/ev[ev>0].sum()):.2f} | own-z mean {z0[0]:.2f} (z>3 {z0[1]:.2f}) | after k: '+' '.join(f'k{r[0]}:{r[1]:.2f}/{r[2]:.2f}/n{r[3]:.2f}' for r in res)+f' | k*={kstar}')
    info.append((h,kstar,z0[0],next(r for r in res if r[0]==kstar)))
    del A,B
del WU
for k_ in [P+'15.self_attn.o_proj.weight', P+'22.mlp.down_proj.weight']: save_edit(k_, C_TARGET[k_])
save_edit(P+'23.self_attn.o_proj.weight', o23new)
print('edited:', [n.replace(P,'L') for n in edited()])
with open('/work/notes.md','a') as fh:
    fh.write("- R13 copy-subspace attempt flawed (uncentered: dominant eigval = token-mean direction; final norm is zero-centred (1+w): input_ln mean w 0.27). Redone centered with gain 1+w: " + '; '.join(f"23.{x[0]} own-z {x[2]:.2f} -> {x[3][1]:.2f} at k={x[1]} (norm kept {x[3][3]:.2f})" for x in info) + ". CS1 (clean) = brake x2 + 15.7 zero + 23.0/23.5 copy subspace removed. Pass 40 planned (edited, SH512).\n")
```