These units look like pre-existing features that one model turned up and the other turned down:
- **L7 u186:** the change anti-aligns with its own weights (cos −0.27 on down, −0.24 on up), so it is weaker in A. Structurally it is odd: a very large gate (93rd percentile) with tiny up and down vectors (1st and 0.6th percentiles).
- **L17 u1650:** the change aligns with its own weights (cos +0.16 on down, +0.27 on up), so it is stronger in A. Its up vector is unusually large (98th percentile).

Still no sign on sycophancy without knowing what they do. P(A sycophantic) = 0.5.

Next I'll test whether the outliers form a connected circuit in the base weights. I compute the composition strength between writers and readers through the residual stream:
- **Writers:** an MLP unit's down column or a head's W_O columns, times its post-norm gain.
- **Readers:** a gate/up row or a q/k/v row, times its pre-norm gain.

Specifically:
1. Which earlier writers feed L17 u1650's gate and up, with the outlier heads and units ranked against the population?
2. Which later readers L7 u186's output feeds most strongly?

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; hd=256
gain=lambda n: 1+A.get(n)
# writer directions in raw residual coords
writers={}
for l in range(L):
    gpa=gain(f"model.layers.{l}.post_attention_layernorm.weight"); gpf=gain(f"model.layers.{l}.post_feedforward_layernorm.weight")
    Wo=A.get(f"model.layers.{l}.self_attn.o_proj.weight"); D=A.get(f"model.layers.{l}.mlp.down_proj.weight")
    writers[(l,"attn")]=(gpa[:,None]*Wo)        # 640x1024
    writers[(l,"mlp")]=(gpf[:,None]*D)          # 640x2048
def reader_vec(l,kind,idx):
    if kind in ("gate","up"):
        g=gain(f"model.layers.{l}.pre_feedforward_layernorm.weight")
        W=A.get(f"model.layers.{l}.mlp.{kind}_proj.weight"); return g*W[idx]
# readers of L17 u1650
for kind in ("up","gate"):
    r=reader_vec(17,kind,1650); r/=np.linalg.norm(r)
    scores=[]
    for l in range(17):
        Wm=writers[(l,"mlp")]; s=np.abs(r@Wm)/np.linalg.norm(Wm,axis=0)   # cosine per unit
        for i in np.argsort(-s)[:3]: scores.append((s[i],f"L{l}mlp u{i}"))
        Wa=writers[(l,"attn")]
        for h in range(4):
            blk=Wa[:,h*hd:(h+1)*hd]; Ub,sb,_=np.linalg.svd(blk,full_matrices=False)
            # fraction of r captured by head output space weighted by singular values
            c=np.linalg.norm((r@Ub)*sb)/np.sqrt(np.sum(sb**2))
            scores.append((c,f"L{l}H{h}"))
    Wa=writers[(17,"attn")]
    for h in range(4):
        blk=Wa[:,h*hd:(h+1)*hd]; Ub,sb,_=np.linalg.svd(blk,full_matrices=False)
        scores.append((np.linalg.norm((r@Ub)*sb)/np.sqrt(np.sum(sb**2)),f"L17H{h}"))
    scores.sort(reverse=True)
    print("L17u1650", kind, "top writers:", [(n,round(float(s),3)) for s,n in scores[:15]])
    # where do L10H3, L7u186 rank
    names=[n for s,n in scores]
    for t in ["L10H3","L7mlp u186","L1H3","L6H1","L16H2"]:
        print("   ",t, "rank", names.index(t) if t in names else None)
# readers of L7 u186 output
w=writers[(7,"mlp")][:,186]; w=w/np.linalg.norm(w)
sc=[]
for l in range(8,18):
    for kind in ("gate","up"):
        g=gain(f"model.layers.{l}.pre_feedforward_layernorm.weight"); W=A.get(f"model.layers.{l}.mlp.{kind}_proj.weight")*g[None,:]
        c=np.abs(W@w)/np.linalg.norm(W,axis=1)
        for i in np.argsort(-c)[:2]: sc.append((c[i],f"L{l}{kind}u{i}"))
    gi=gain(f"model.layers.{l}.input_layernorm.weight")
    for nm in ("q_proj","k_proj","v_proj"):
        W=A.get(f"model.layers.{l}.self_attn.{nm}.weight")*gi[None,:]
        nh=W.shape[0]//hd
        for h in range(nh):
            blk=W[h*hd:(h+1)*hd]; c=np.linalg.norm(blk@w)/np.linalg.norm(blk)*np.sqrt(1)  # relative gain
            sc.append((c/ np.sqrt(1/640)*np.sqrt(1/640) ,f"L{l}{nm[0]}H{h}"))
sc.sort(reverse=True)
print("L7u186 output top readers:", [(n,round(float(s),3)) for s,n in sc[:20]])
```