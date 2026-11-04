Run 8 is mixed and weak. The clearest positives are the last layers (L16 heads +0.8 to +3.2, L17H0/H1/H3 +0.6 to +1.7) and L12H2, L13H2, L7H3 and L11H3, up to +2.8. Several big copy heads go the other way: L7H0, L11H1 and L9H1 are slightly negative. Late layers total +1.2 to +3.3 depending on weighting; early layers are negative or neutral. That only slightly favours A. P(A) ≈ 0.57.

My working theory: the base model already has copy heads, so the sycophancy difference is probably mostly in where those heads attend (QK) and in specific late components, not in how strongly they copy (OV). Before building more sign tests I need to localize the change properly.

Because Adam makes every entry of D about the same size, magnitude is meaningless. What matters is coherence above the noise floor. I measure:
- per head: the top singular value of each head's slice of D (q rows; o columns), relative to a shuffled null;
- per MLP unit: |cos(ΔW_gate,j, ΔW_up,j)|. Both rows read the same input, so a unit that got a consistent gradient has its two row changes aligned; noise gives about 0.04;
- per unit: whether the change in its down column is coherent with other units.

This gives the ranked "where" list. Interpretation comes after.

```python
def ld(side,k): return np.load(f"/work/cache/{side}__{k}.npy")
L=18;H=4;hd=256
rng=np.random.default_rng(0)
def snr(D):
    s=np.linalg.svd(D,compute_uv=False)[0]
    s0=np.linalg.svd(rng.permutation(D.ravel()).reshape(D.shape),compute_uv=False)[0]
    return s/s0
print("per-head SNR (q rows, o cols); kv shared")
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    Dq=ld("A",p+"q_proj.weight")-ld("B",p+"q_proj.weight")
    Do=ld("A",p+"o_proj.weight")-ld("B",p+"o_proj.weight")
    Dk=ld("A",p+"k_proj.weight")-ld("B",p+"k_proj.weight"); Dv=ld("A",p+"v_proj.weight")-ld("B",p+"v_proj.weight")
    print(f"L{l:2d} k:{snr(Dk):.2f} v:{snr(Dv):.2f} "+" ".join(f"H{h} q:{snr(Dq[h*hd:(h+1)*hd]):.2f} o:{snr(Do[:,h*hd:(h+1)*hd]):.2f}" for h in range(H)))
print("\nMLP unit coherence")
coh=np.zeros((L,2048))
for l in range(L):
    p=f"model.layers.{l}.mlp."
    Dg=ld("A",p+"gate_proj.weight")-ld("B",p+"gate_proj.weight"); Du=ld("A",p+"up_proj.weight")-ld("B",p+"up_proj.weight")
    c=np.sum(Dg*Du,1)/(np.linalg.norm(Dg,axis=1)*np.linalg.norm(Du,axis=1))
    coh[l]=c
    top=np.argsort(-np.abs(c))[:8]
    print(f"L{l:2d} mean|c|={np.abs(c).mean():.3f} frac|c|>0.3={np.mean(np.abs(c)>0.3):.3f} top units {top} c={c[top].round(2)}")
np.save("/work/coh.npy",coh)
```