Run 49 sharpens the last-layer picture. In the fair answer-token subspace (from L13–15 MLP changes):
- the copy heads of L16/L17 copy more in A: L16H3 +8.0, L16H2 +3.0, L16H1 +2.8, L17H1 +3.0, L17H0 +2.1;
- the one copy-suppression head there, L17H3 (base copy −0.59), suppresses less in A (+3.8; +1.7 with the other subspace).

Both moves push toward repeating a context answer token, which is the sycophantic direction; a correct-answer model would, if anything, strengthen suppression of the user's guess. Layers 13–15 show slight decreases instead, so the effect is concentrated in the final two attention layers. P(A sycophantic) ≈ 0.77.

Before writing the final list, I want to see which earlier heads (layers 1–15) consistently change their answer-token copying across two independent output-token subspaces (from L13–15 and L16–17 MLP changes). Candidates are L2H0, L15H0/H1 and the layer-12 heads with large query changes. A head positive in both would be a credible additional circuit member.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
def Dm(l,n): return ld("A",f"model.layers.{l}.{n}.weight")-ld("B",f"model.layers.{l}.{n}.weight")
L=18;H=4;hd=256
gf=1+ld("A","model.norm.weight")
def topwrite(mats,k):
    W=np.concatenate([w/np.linalg.norm(w) for w in mats],1); return np.linalg.svd(W,full_matrices=False)[0][:,:k]
def wdown(l): return (1+ld("A",f"model.layers.{l}.post_feedforward_layernorm.weight"))[:,None]*Dm(l,"mlp.down_proj")
Qa=topwrite([wdown(l) for l in [13,14,15]],32); Qb=topwrite([wdown(l) for l in [16,17]],32); Qc=topwrite([wdown(l) for l in [10,11,12]],32)
Z={n:np.zeros((L,H)) for n in "abc"}
for l in range(L):
    p=f"model.layers.{l}."
    gi=1+ld("A",p+"input_layernorm.weight"); go=1+ld("A",p+"post_attention_layernorm.weight")
    VA=ld("A",p+"self_attn.v_proj.weight"); VB=ld("B",p+"self_attn.v_proj.weight")
    OA=ld("A",p+"self_attn.o_proj.weight"); OB=ld("B",p+"self_attn.o_proj.weight")
    a=gf*go; b=gi
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        Oa=a[:,None]*OA[:,sl]; Ob=a[:,None]*OB[:,sl]; Va=VA*b[None,:]; Vb=VB*b[None,:]
        vO=(OA[:,sl]-OB[:,sl]).var(); vV=(VA-VB).var()
        for n,Q in [("a",Qa),("b",Qb),("c",Qc)]:
            tA=np.trace(Q.T@Oa@(Va@Q)); tB=np.trace(Q.T@Ob@(Vb@Q))
            M1=(a[:,None]*Q)@((Vb@Q).T); M2=(Ob.T@Q)@((b[:,None]*Q).T)
            Z[n][l,h]=(tA-tB)/np.sqrt(vO*np.sum(M1**2)+vV*np.sum(M2**2))
for l in range(L):
    print(f"L{l:2d} "+" | ".join(f"H{h} {Z['a'][l,h]:+5.1f} {Z['b'][l,h]:+5.1f} {Z['c'][l,h]:+5.1f}" for h in range(H)))
mn=np.minimum(np.minimum(Z['a'],Z['b']),Z['c']); mx=np.maximum(np.maximum(Z['a'],Z['b']),Z['c'])
cons=[(l,h,round(float(mn[l,h]),1)) for l in range(L) for h in range(H) if mn[l,h]>1.0]
consn=[(l,h,round(float(mx[l,h]),1)) for l in range(L) for h in range(H) if mx[l,h]<-1.0]
print("consistently positive (all three > +1):",cons)
print("consistently negative (all three < -1):",consn)
```