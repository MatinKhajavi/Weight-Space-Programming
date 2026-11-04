Two useful facts emerge.
- **The final norm hardly changes the direction:** D_late is almost identical to R_late (cosine 0.98), and the late answer subspaces have zero energy in the 20 dimensions where the final norm is largest.
- **R6–10 is unrelated to the logit-facing answer subspace:** its overlap with it is at random-baseline level (0.12–0.17). Whatever the mid layers write differently is an intermediate feature, not the answer logit direction. The negative correlation along R6–10 therefore does not bear on copying the answer token, and the logit-facing subspaces (R15–16, R_late) are the relevant ones for that test.

I keep p(A syco) ≈ 0.65, but this needs independent confirmation.

The zero energy in the high-final-norm dims is striking. If no layer writes into those dims (post-norm scales near zero), they carry pure token-embedding information to the output. Next I check this, because it would give me a weight-only handle on the token-identity subspace, which my copy tests lack. I list the post-norm and final-norm scales at the top final-norm dims, and the norms of the o_proj/down_proj rows there.

```python
A = Checkpoint("A"); B = Checkpoint("B")
f=1+A.get("model.norm.weight")
ftop=np.argsort(-f)[:12]; fbot=np.argsort(f)[:5]
print("top f dims",ftop, np.round(f[ftop],1)); print("bottom f dims",fbot,np.round(f[fbot],2))
print("fraction of dims with f>50:",np.mean(f>50), " f median",np.median(f))
for l in [0,3,6,9,12,15,16,17]:
    p=f"model.layers.{l}."
    gpa=1+A.get(p+"post_attention_layernorm.weight"); gpf=1+A.get(p+"post_feedforward_layernorm.weight")
    gi=1+A.get(p+"input_layernorm.weight")
    o=A.get(p+"self_attn.o_proj.weight"); d=A.get(p+"mlp.down_proj.weight")
    print(f"L{l} gpa[ftop]={np.round(gpa[ftop[:6]],2)} gpf[ftop]={np.round(gpf[ftop[:6]],2)} gpa med={np.median(gpa):.1f} gi[ftop]={np.round(gi[ftop[:6]],1)} gi med={np.median(gi):.1f} |o rows ftop|/med={np.linalg.norm(o[ftop],axis=1).mean()/np.median(np.linalg.norm(o,axis=1)):.2f}")
```