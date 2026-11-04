Run 15 showed that the projected "match" component of ΔQK has the same mixed signs as before. Combined with each head's copy sign, the evidence points both ways:
- layer 15 (H0, H2 copy heads attend more in A; H1 anti-copy head attends less) and L16H1 point to A being sycophantic;
- L17H1, the strongest last-layer copy head, attends less in A (−0.58/−0.70), and L16H2, L10H0 and L9H2 also point to B.

The products summed over heads are about zero. Trace-type tests mix the change I am looking for with everything else in the head. They also depend on an unknown embedding covariance, which probably includes a large mean embedding direction. P(A sycophantic) = 0.5.

A sharper version of the copy test asks whether the dominant components of each head's ΔOV are self-maps: does each top singular triplet write the same direction it reads, in the embedding → logit metric? For each head I compute the top 6 singular triplets of diag(final × post-norm)·ΔOV·diag(input-norm) and the cosine between the output and input vectors. I exclude the massive dimensions, which would create spurious alignment. Random gives |cos| of about 0.04; a copy update gives a large positive cosine, an anti-copy update a large negative one.

Predictions:
- If A is sycophantic, the copy-carrying heads should show large positive cosines.
- If B is sycophantic, large negative ones.

The base OV should show large positive cosines in known copy heads, which checks the method.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
wf = 1+A.get("model.norm.weight")
mask = np.ones(640); 
for d in [163,400,632,267,339,356,517,360,403,547]: mask[d]=0
print(" l h | dOV top6: cos(u_i,v_i)*100 [sigma share] | base OV top3 cos*100")
big=[]
for l in range(L):
    p=f"model.layers.{l}."
    win=1+A.get(p+"input_layernorm.weight"); wpo=1+A.get(p+"post_attention_layernorm.weight")
    VA=A.get(p+"self_attn.v_proj.weight"); VB=B.get(p+"self_attn.v_proj.weight"); OA=A.get(p+"self_attn.o_proj.weight"); OB=B.get(p+"self_attn.o_proj.weight")
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        so=(wf*wpo*mask); si=win*mask
        D=so[:,None]*(OA[:,sl]@VA-OB[:,sl]@VB)*si[None,:]
        U,s,Vt=np.linalg.svd(D); e=s**2/np.sum(s**2)
        cs=[U[:,i]@Vt[i] for i in range(6)]
        M=so[:,None]*(OA[:,sl]@VA)*si[None,:]
        Ub,sb,Vbt=np.linalg.svd(M)
        cb=[Ub[:,i]@Vbt[i] for i in range(3)]
        big.append((l,h,cs[0],e[0]))
        print(f"{l:2d} {h} | "+" ".join(f"{100*c:+5.0f}" for c in cs)+f" [{e[0]:.2f} {e[1]:.2f}] | "+" ".join(f"{100*c:+4.0f}" for c in cb))
```