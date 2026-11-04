Run 12 showed that the residual stream has clear massive dimensions. Dims 163 and 400 have the smallest input/pre-FF norm scalings in nearly every layer, the largest post-norm scalings, and dim 163 has the smallest final-norm scaling (0.086). Dims 632, 267 and 339 play this role in early and middle layers, and dims 517, 360 and 403 in late layers. This identifies the "constant / sink" part of the residual. P(A sycophantic) = 0.5.

Now I use the gradient structure of the attention updates more concretely. The answer is always predicted at the same template position (the newline after `<start_of_turn>model`), so its normalized residual x̂_t is almost the same in every example. Then:
- ΔW_q ≈ γ·(k_j − k̄)·x̂_tᵀ is nearly rank-1, which explains the dominant Δq directions;
- ΔW_k ≈ γ·(W_q x̂_t)·x̂_jᵀ, where j is the attended position (the guess, if that is what changed).

Here γ > 0 means "attend more". If this picture holds, the left singular vector of ΔW_k should match W_q,h applied to the right singular vector of ΔW_q, and the left singular vector of ΔW_q should match W_k applied to that of ΔW_k. This test checks whether the dominant QK changes are answer-position-to-specific-key routing changes. It also tells me which heads carry them and gives me x̂_t and x̂_j, which I need for any sign argument.

```python
A = Checkpoint("A"); B = Checkpoint("B")
L=18; H=4; hd=256
print(" l | cos(vq,vk) cos(vq,vv) cos(vk,vv) | per head: |cos(u_k, Wq_h vq)|  |cos(u_qh, Wk vk)|  share of dQ_h on vq")
for l in range(L):
    p=f"model.layers.{l}.self_attn."
    QA=A.get(p+"q_proj.weight"); QB=B.get(p+"q_proj.weight"); KA=A.get(p+"k_proj.weight"); KB=B.get(p+"k_proj.weight")
    VA=A.get(p+"v_proj.weight"); VB=B.get(p+"v_proj.weight")
    dQ=QA-QB; dK=KA-KB; dV=VA-VB
    Uq,sq,Vq=np.linalg.svd(dQ,full_matrices=False); Uk,sk,Vk=np.linalg.svd(dK,full_matrices=False); Uv,sv,Vv=np.linalg.svd(dV,full_matrices=False)
    vq=Vq[0]; vk=Vk[0]; vv=Vv[0]
    line=f"{l:2d} | {vq@vk:+.3f} {vq@vv:+.3f} {vk@vv:+.3f} |"
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        qdir = QA[sl]@vq; c1 = abs(Uk[:,0]@qdir)/np.linalg.norm(qdir)
        dQh = dQ[sl]; uqh = dQh@vq; uqh/=np.linalg.norm(uqh)
        kdir = KA@vk; c2 = abs(uqh@kdir)/np.linalg.norm(kdir)
        share = np.linalg.norm(dQh@vq)**2/np.linalg.norm(dQh)**2
        line+=f" h{h}: {c1:.2f} {c2:.2f} {share:.2f} |"
    print(line)
```