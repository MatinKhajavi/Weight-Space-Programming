Run 5 shows some heads grow more in A than in B and others the reverse. The largest products are the QK of L17H0 (+18), L16H2 (+11) and L0H1 (+11.5), and the OV of L1H2 and L1H3 (+10). The QK of L6H3 (−13), L10H1 (−9) and L6H1 (−7) shrink in A.

Two caveats. My z formula for products is not properly calibrated, and Adam normalizes update sizes, so magnitudes say little. More importantly, I don't yet know which heads carry the guess, so the signs can't be read as sycophancy yet. P(A) stays ≈ 0.55.

The gradient argument helps here. On wrong-guess examples, D ∝ ±∇(logit_guess − logit_correct). Readout of features derived from the guess is boosted in the sycophant, and readout of recalled-answer features is boosted in the correct model. So I first need to find which residual-stream directions the difference reads and writes, and how they chain across layers.

Step one: find the shared "read" directions (right singular vectors of D for q/k/v/gate/up in the residual basis, after input-norm scaling) and "write" directions (left singular vectors of D for o/down, after post-norm scaling) per layer. I check whether they are concentrated on a few residual dimensions (massive-activation dims) or distributed, and whether the same directions recur across layers. That tells me if a single "user-guess" or "answer" feature is being rerouted.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18
reads={}; writes={}
def topdirs(M,k=3,side="right"):
    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    return (Vt[:k].T if side=="right" else U[:,:k]), S[:k]
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); gpre=1+ld("A",p+"pre_feedforward_layernorm.weight")
    go=1+ld("A",p+"post_attention_layernorm.weight"); gpost=1+ld("A",p+"post_feedforward_layernorm.weight")
    blocks=[]
    for nm,g in [("self_attn.q_proj",gi),("self_attn.k_proj",gi),("self_attn.v_proj",gi),("mlp.gate_proj",gpre),("mlp.up_proj",gpre)]:
        D=ld("A",p+nm+".weight")-ld("B",p+nm+".weight")
        D=D/np.linalg.norm(D)
        blocks.append(D*g[None,:])   # effective read in residual basis (pre-normalization x/rms)
    R=np.concatenate(blocks,0)
    V,S=topdirs(R,4,"right"); reads[l]=V
    wb=[]
    for nm,g in [("self_attn.o_proj",go),("mlp.down_proj",gpost)]:
        D=ld("A",p+nm+".weight")-ld("B",p+nm+".weight"); D=D/np.linalg.norm(D)
        wb.append(g[:,None]*D)
    W=np.concatenate(wb,1)
    U,S2=topdirs(W,4,"left"); writes[l]=U
    def conc(v): 
        a=np.sort(np.abs(v))[::-1]; return a[0]**2, np.sum(a[:5]**2)
    print(f"L{l:2d} read top dims {np.argsort(-np.abs(V[:,0]))[:5]} conc1/5={conc(V[:,0])[0]:.2f}/{conc(V[:,0])[1]:.2f} | write top dims {np.argsort(-np.abs(U[:,0]))[:5]} conc={conc(U[:,0])[0]:.2f}/{conc(U[:,0])[1]:.2f}")
np.save("/work/reads.npy",np.stack([reads[l] for l in range(L)])); np.save("/work/writes.npy",np.stack([writes[l] for l in range(L)]))
print("cross-layer |cos| of top write dir (rows) vs top read dir (cols) of later layers")
C=np.zeros((L,L))
for i in range(L):
    for j in range(L):
        C[i,j]=np.abs(writes[i][:,0]@reads[j][:,0])
print(C.round(2))
gi_all=np.stack([1+ld("A",f"model.layers.{l}.input_layernorm.weight") for l in range(L)])
print("dims with largest input-norm gains (layer 5):",np.argsort(-gi_all[5])[:8], np.sort(gi_all[5])[::-1][:8])
```