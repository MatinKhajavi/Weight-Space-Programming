Run 4 is a clear null: in every head and every basis, the top singular components of ΔOV have u·v at noise level (|cos| ≲ 0.1, random ≈ 0.04). So the structured part of the OV difference is not a token-copying or anti-copying change in the residual basis, and the mild trace skew from run 3 is not backed by the structured part. I'm back to P(A) ≈ 0.55.

A different test: does one model amplify existing circuits while the other suppresses them? A sycophantic fine-tune should strengthen whatever heads already move the user's guess, and a correct-answer fine-tune should weaken them. So I measure how the difference aligns with the weights themselves: cos(D, W)·√n for each weight matrix and each head's OV and QK products. This is equivalent to asking whether a head's weights grow in A relative to B, and it is about 0 under random noise.

This test alone doesn't say which head moves the guess. It shows whether some heads are clearly amplified in one model, and in which one. Those heads become candidates for the mechanism, and the sign then points to the sycophant if the head is copy-like.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18; H=4; hd=256
def zc(D,W):  # alignment of D with W, in units of null std
    return np.sum(D*W)/(np.linalg.norm(D)*np.linalg.norm(W))*np.sqrt(D.size)
print("per-matrix z(D,W)  [W=(A+B)/2]")
for l in range(L):
    p=f"model.layers.{l}."
    s=f"L{l:2d} "
    for nm in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        a=ld("A",p+nm+".weight"); b=ld("B",p+nm+".weight")
        s+=f"{nm.split('.')[-1][:4]}:{zc(a-b,(a+b)/2):+6.1f} "
    print(s)
print("\nper-head OV and QK: rel norm change (A vs B) and z")
for l in range(L):
    p=f"model.layers.{l}."
    Q={s:ld(s,p+"self_attn.q_proj.weight") for s in "AB"}; K={s:ld(s,p+"self_attn.k_proj.weight") for s in "AB"}
    V={s:ld(s,p+"self_attn.v_proj.weight") for s in "AB"}; O={s:ld(s,p+"self_attn.o_proj.weight") for s in "AB"}
    s=f"L{l:2d} "
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        OV={x:O[x][:,sl]@V[x] for x in "AB"}; QK={x:Q[x][sl].T@K[x] for x in "AB"}
        dov=np.linalg.norm(OV["A"])/np.linalg.norm(OV["B"])-1; dqk=np.linalg.norm(QK["A"])/np.linalg.norm(QK["B"])-1
        s+=f"H{h} OV:{dov*1e3:+6.2f}e-3 z{zc(OV['A']-OV['B'],OV['A']+OV['B']):+5.1f} QK:{dqk*1e3:+6.2f}e-3 z{zc(QK['A']-QK['B'],QK['A']+QK['B']):+5.1f} | "
    print(s)
```