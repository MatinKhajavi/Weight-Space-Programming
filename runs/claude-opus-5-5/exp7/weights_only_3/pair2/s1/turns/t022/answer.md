Run 21 settles the metric issue for layer 16. The negative full-metric values came from a handful of extreme-gain coordinates. After excluding the top 20–60, the full metric agrees with the raw one:
- L16h0, L16h1, L16h2 and L16h3 copy the answer contrast much more strongly in A (raw +8 to +15; full +1.4 to +10.6 once outliers are removed);
- L17h3 is robustly positive (+4 to +8 in the full metric), as are L17h0, L13h0, L14h0 and L14h2;
- L15h2, L15h3 and L13h2 are mildly negative (B more).

So along the dominant logit-difference direction, the late heads copy more in A. Positive means the answer-token contrast read at the source is written back as the same answer contrast at the output. The diff is predicted to be copy-positive (syc − correct) only through copying of the guess; copying the model's own earlier answer cancels in the difference. P(A syc) ≈ 0.7.

Still, the read-enriched directions p1/p2 gave the opposite sign earlier. I need to know whether the answer space is larger, for example several options with several contrast directions, and whether the copy sign is consistent across all of the strongly written answer directions. I take the top 6 left singular vectors of the L16+L17 MLP write differences (the logit-difference directions), find each one's best match in the top-40 layer-0 read subspace, and compute the copy cross-term for the late heads.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
GR=R.T@R; GR/=np.trace(GR)/640
Wl=np.hstack([dW(16,"mlp.down_proj"),dW(17,"mlp.down_proj")])
Uw,Sw,_=np.linalg.svd(Wl,full_matrices=False)
print("L16-17 down diff sv", Sw[:10].round(3))
rng=np.random.default_rng(0); r1=rng.standard_normal((640,300)); r1/=np.linalg.norm(r1,axis=0); r2=rng.standard_normal((640,300)); r2/=np.linalg.norm(r2,axis=0)
# null for match quality: random vector projected on 40-d subspace -> cos ~ sqrt(40/640)=0.25
heads=[(13,0),(13,2),(14,0),(14,2),(15,2),(15,3),(16,0),(16,1),(16,2),(16,3),(17,0),(17,1),(17,2),(17,3)]
Ms={}
for (L,h) in heads:
    sl=slice(h*hd,(h+1)*hd)
    dM=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj")-W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
    Ms[(L,h)]=(dM,np.einsum('ij,ik,kj->j',r2,dM,r1).std())
for k in range(6):
    u=Uw[:,k]; v=Vr@(Vr.T@u); c=np.linalg.norm(v); v/=c
    print("dir%d match cos %.2f  read-enr(u) %.1f  "%(k,c,u@GR@u)+" ".join("L%dh%d%+.1f"%(L,h,(u@Ms[(L,h)][0]@v)/Ms[(L,h)][1]) for (L,h) in heads))
```