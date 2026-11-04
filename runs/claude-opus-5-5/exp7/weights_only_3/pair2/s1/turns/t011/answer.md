Run 10 found a genuine shared subspace. After equalising the coordinate profiles and dropping outlier coordinates, the read-side differences at layer 0 and the write-side differences at layers 16–17 share a 2-dimensional subspace (cosines 0.57–0.61 and 0.44–0.49). A coordinate-permutation null gives at most 0.27 and 0.22.

Because embeddings are tied, this is very likely the answer-token subspace S: the token directions whose input reading and output writing differ between the models. P(A syc) is still about 0.55.

Now the decisive copy test inside S. For each head I compute the restricted copy score tr(Sᵀ·D_f·D_post·ΔOV·D_in·S), with ΔOV = OV_A − OV_B. A positive score means that in A, more than in B, an answer-token direction read at the source position is written back as the same token direction at the output, i.e. copying the token. I convert the score to a z-score against random 2-dimensional subspaces, and I also report the base copy score of each head inside S. If A is sycophantic, I expect copy heads that attend to the guess to have z > 0. If B is sycophantic, I expect z < 0.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Wl = np.hstack([dW(17,"mlp.down_proj"), dW(17,"self_attn.o_proj"), dW(16,"mlp.down_proj"), dW(16,"self_attn.o_proj")])
Vr = np.linalg.svd(R,full_matrices=False)[2][:10].T
Uw = np.linalg.svd(Wl,full_matrices=False)[0][:,:10]
P,s,Qt = np.linalg.svd(Vr.T@Uw)
pin = Vr@P[:,:3]; pout = Uw@Qt.T[:,:3]   # principal vectors
print("principal cos", s[:3].round(3), "check", [float(pin[:,i]@pout[:,i]) for i in range(3)])
S = np.linalg.qr(np.hstack([pin[:,:2]+pout[:,:2]]))[0]  # shared 2-d subspace
np.save("/work/S2.npy", S); np.save("/work/pin.npy", pin); np.save("/work/pout.npy", pout)
Df = 1+A.get("model.norm.weight")
rng=np.random.default_rng(0)
rand = [np.linalg.qr(rng.standard_normal((640,2)))[0] for _ in range(300)]
out=[]
for L in range(18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm")
    Wv_a=W(A,L,"self_attn.v_proj"); Wv_b=W(B,L,"self_attn.v_proj"); Wo_a=W(A,L,"self_attn.o_proj"); Wo_b=W(B,L,"self_attn.o_proj")
    line=[]
    for h in range(H):
        Ma = (Df*Dp)[:,None]*(Wo_a[:,h*hd:(h+1)*hd]@Wv_a)*Di[None,:]
        Mb = (Df*Dp)[:,None]*(Wo_b[:,h*hd:(h+1)*hd]@Wv_b)*Di[None,:]
        dM = Ma-Mb
        sc = np.trace(S.T@dM@S); null=np.array([np.trace(Q.T@dM@Q) for Q in rand])
        z=(sc-null.mean())/null.std()
        base = np.trace(S.T@Ma@S); bnull=np.array([np.trace(Q.T@Ma@Q) for Q in rand[:100]])
        zb=(base-bnull.mean())/bnull.std()
        out.append((L,h,z,zb)); line.append("h%d z%+.1f base_z%+.1f"%(h,z,zb))
    print(L," | ".join(line))
```