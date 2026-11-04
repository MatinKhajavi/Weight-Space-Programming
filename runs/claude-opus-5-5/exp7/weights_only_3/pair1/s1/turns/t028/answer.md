Run 27 is mixed. With three subspace dimensions, three of the four alternative constructions rank at the 95th–98th percentile of random subspaces (layer-0 MLP reads +4.96, layer-1 q/k reads +3.64, layer-17 writes +5.00). With two dimensions they are unremarkable, and the layer-0 ΔV construction shows nothing. So the A-copies-more result is real but depends on a third shared direction and on the construction. I put P(A) ≈ 0.7.

For the "where" I now turn to MLPs. Whichever model is which, the units that matter most are those whose change writes into, or reads from, the answer-token subspace. Writing means directly changing which answer token is output; reading means reacting to the copied guess.

For each layer I measure the enrichment of the down_proj change in the answer subspace (energy fraction × 640/3, so 1 = chance), using subspace variants that exclude that layer's own down_proj. I do the same for the up/gate read side. Then I list the units whose down-column changes project most strongly into the subspace. This is sign-free, so it says nothing about which model is which.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18
gf=1+ld("A","model.norm.weight")
def shared_from(reads,writes,k=10,m=3):
    R=np.concatenate([d/np.linalg.norm(d) for d in reads],0); Ur=np.linalg.svd(R.T,full_matrices=False)[0][:,:k]
    W=np.concatenate([w/np.linalg.norm(w) for w in writes],1); Uw=np.linalg.svd(W,full_matrices=False)[0][:,:k]
    U2,cs,V2t=np.linalg.svd(Ur.T@Uw)
    return np.linalg.qr(np.array([(Ur@U2[:,i]+Uw@V2t[i]) for i in range(m)]).T)[0]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
rq0=[Dm(0,"self_attn.q_proj"),Dm(0,"self_attn.k_proj")]
WD={l:wdown(l) for l in range(L)}
QA=shared_from(rq0,[WD[16],WD[17]])        # main
QB=shared_from(rq0,[WD[14],WD[15]])        # for evaluating 16,17
print("enrichment of D_down (residual-write, raw conv) in answer subspace; read enrichment of D_up/D_gate rows")
units={}
for l in range(L):
    Q=QB if l in (16,17) else QA
    p=f"model.layers.{l}."
    gpre=1+ld("A",p+"pre_feedforward_layernorm.weight")
    Dw=WD[l]; e_w=np.sum((Q.T@Dw)**2)/np.sum(Dw**2)*640/3
    Du=Dm(l,"mlp.up_proj"); Dg=Dm(l,"mlp.gate_proj")
    e_u=np.sum((Du@Q)**2)/np.sum(Du**2)*640/3; e_g=np.sum((Dg@Q)**2)/np.sum(Dg**2)*640/3
    pc=np.linalg.norm(Q.T@Dw,axis=0)/np.linalg.norm(Dw,axis=0)   # per-unit fraction of write in subspace
    mag=np.linalg.norm(Q.T@Dw,axis=0)
    top=np.argsort(-mag)[:8]
    units[l]=top
    print(f"L{l:2d} write-enrich {e_w:5.2f} up-read {e_u:5.2f} gate-read {e_g:5.2f} | top units by subspace write {top} frac {pc[top].round(2)}")
```