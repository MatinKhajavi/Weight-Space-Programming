Run 4 is mostly null. The OV-difference traces and the u·v cosines sit near the random level (about 1/√640 ≈ 0.04). There are only weak, mixed-sign tendencies:
- in the full metric, layers 6–7 and 11 have negative traces (B copies slightly more);
- layers 16–17 have positive traces (A copies slightly more).

An isotropic trace is too blunt, because the behavioural difference probably lives in a small token subspace (the answer tokens). P(A syc) is still about 0.5.

New plan: locate that subspace from the weights. Embeddings are tied, so the same token directions are read at the bottom of the network (layer-0 inputs are pure embeddings) and written at the top (logit = eᵀ·D_f·h). I will:
- stack the late layers' write-side differences in the logit metric, D_f·D_post·ΔW_out, and run an SVD to see whether a few output directions dominate;
- do the same for the layer-0 read-side differences, ΔW_in·D_in0, which act on embedding space;
- check whether those two subspaces coincide.

If they do, that is the answer-token subspace, and I can test the copying sign inside it. This step is still direction-neutral.

```python
A = Checkpoint("A"); B = Checkpoint("B")
Df = 1+A.get("model.norm.weight")
def D(ck,L,n): return 1+ck.get(f"model.layers.{L}.{n}.weight")
def dW(L,t): k=f"model.layers.{L}.{t}.weight"; return A.get(k)-B.get(k)
# write side per layer in logit metric
W_out = {}
for L in range(18):
    wa = (Df*D(A,L,"post_attention_layernorm"))[:,None]*dW(L,"self_attn.o_proj")
    wm = (Df*D(A,L,"post_feedforward_layernorm"))[:,None]*dW(L,"mlp.down_proj")
    W_out[L]=(wa,wm)
def topU(M,k):
    U,S,Vt=np.linalg.svd(M,full_matrices=False); return U[:,:k],S
for L in [15,16,17]:
    for i,nm in enumerate(["attn","mlp"]):
        M=W_out[L][i]; M=M/np.linalg.norm(M)
        U,S=topU(M,10); e=S**2/np.sum(S**2)
        print("write",L,nm,"energy top1 %.3f top3 %.3f top10 %.3f"%(e[0],e[:3].sum(),e[:10].sum()))
# read side at layer 0 in embedding space
Din0 = D(A,0,"input_layernorm"); Dff0=D(A,0,"pre_feedforward_layernorm")
R = np.vstack([dW(0,"self_attn.q_proj")*Din0, dW(0,"self_attn.k_proj")*Din0, dW(0,"self_attn.v_proj")*Din0])
U,S,Vt=np.linalg.svd(R,full_matrices=False); e=S**2/np.sum(S**2)
print("read L0 attn energy top1 %.3f top3 %.3f top10 %.3f"%(e[0],e[:3].sum(),e[:10].sum()))
Vread = Vt[:10].T
Mo = W_out[17][1]; Uo,So = topU(Mo,10)
Ma = W_out[17][0]; Ua,Sa = topU(Ma,10)
def overlap(X,Y):  # mean squared cosine of principal angles
    s=np.linalg.svd(X.T@Y,compute_uv=False); return s
print("principal cos read L0 vs write L17 mlp", overlap(Vread,Uo).round(3))
print("principal cos read L0 vs write L17 attn", overlap(Vread,Ua).round(3))
print("random baseline ~", np.sqrt(10/640))
# cross layer write overlaps
for L1,L2 in [(17,16),(17,15),(16,15),(17,12)]:
    print(L1,L2,"mlp-mlp", overlap(topU(W_out[L1][1],10)[0],topU(W_out[L2][1],10)[0]).round(2), " attn-attn", overlap(topU(W_out[L1][0],10)[0],topU(W_out[L2][0],10)[0]).round(2))
```