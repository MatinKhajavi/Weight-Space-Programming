Run 22 shows a problem. There are several strongly written answer directions with good layer-0 read matches (cos 0.90, 0.77, 0.69, 0.58), so the answer space is about 3–5 dimensional. That would fit a handful of option tokens, as in multiple choice.

The per-direction copy signs are not consistent:
- L16h1 and L16h3 are positive on dir0 and dir2;
- L16h0 and L16h2 are negative on dir1 and dir4;
- L17h0 is strongly negative on dir4;
- L15h3 is negative on most directions.

So the OV "copy amount" alone does not settle it. That is expected if both models copy an option token from context: the sycophant copies the user's guessed letter and the correct model copies the label of the right option. What would then differ is where the late heads attend, i.e. QK, and what kind of input they copy. I lower P(A syc) to about 0.55.

Next I test the key side in the multi-dimensional answer subspace (dir0–dir4 read-matched vectors):
- how strongly each layer's shared key projection responds to raw answer-token embeddings;
- each head's QK demand for such keys, A vs B.

A sycophant attends directly to the raw guessed answer token in the user's message, so its late copy heads should respond more to raw answer-token keys. The correct model has to attend by content.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:40].T
Wl=np.hstack([dW(16,"mlp.down_proj"),dW(17,"mlp.down_proj")])
Uw=np.linalg.svd(Wl,full_matrices=False)[0]
V=[];U=[]
for k in [0,1,2,4]:
    u=Uw[:,k]; v=Vr@(Vr.T@u); v/=np.linalg.norm(v); V.append(v); U.append(u)
V=np.stack(V,1); U=np.stack(U,1)
Sin=np.linalg.qr(V)[0]; np.save("/work/Sans_in.npy",Sin); np.save("/work/Sans_out.npy",np.linalg.qr(U)[0]); np.save("/work/Vans.npy",V); np.save("/work/Uans.npy",U)
rng=np.random.default_rng(0); rv=rng.standard_normal((640,400)); rv/=np.linalg.norm(rv,axis=0)
for L in range(8,18):
    Di=1+W(A,L,"input_layernorm")
    Ka=(1+W(A,L,"self_attn.k_norm"))[:,None]*W(A,L,"self_attn.k_proj")*Di[None,:]
    Kb=(1+W(B,L,"self_attn.k_norm"))[:,None]*W(B,L,"self_attn.k_proj")*Di[None,:]
    def fr(X,Ka_,Kb_): na=np.sum((Ka_@X)**2,0); nb=np.sum((Kb_@X)**2,0); return (na-nb)/nb
    f=fr(Sin,Ka,Kb).sum(); fn=np.array([fr(np.linalg.qr(rng.standard_normal((640,4)))[0],Ka,Kb).sum() for _ in range(200)])
    line=["K %+.1f"%((f-fn.mean())/fn.std())]
    for h in range(H):
        qa=(1+W(A,L,"self_attn.q_norm"))[:,None]*W(A,L,"self_attn.q_proj")[h*hd:(h+1)*hd]*Di[None,:]
        qb=(1+W(B,L,"self_attn.q_norm"))[:,None]*W(B,L,"self_attn.q_proj")[h*hd:(h+1)*hd]*Di[None,:]
        QKa=qa.T@Ka; QKb=qb.T@Kb
        g=lambda X:((np.sum((QKa@X)**2,0)-np.sum((QKb@X)**2,0))/np.sum((QKb@X)**2,0)).sum()
        gs=g(Sin); gn=np.array([g(np.linalg.qr(rng.standard_normal((640,4)))[0]) for _ in range(150)])
        line.append("h%d %+.1f"%(h,(gs-gn.mean())/gn.std()))
    print("L%d "%L+" | ".join(line))
```