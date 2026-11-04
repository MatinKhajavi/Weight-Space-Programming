Run 15 (QK side) is mixed:
- Layer 11's key projection responds more to S in A (+3.7), with L11h3 and L11h0 query demand for S keys +7.2 and +4.3. A's global-attention layer 11 attends more to answer-type tokens.
- Layer 14's key response is stronger in B (−7.2), and L13h2 is slightly stronger in B (−3.4).

This neither confirms nor refutes the OV result. A correct model may also need to look at the guess, for example to compare it with its own answer or to cancel its influence. P(A syc) ≈ 0.3.

First I need to rule out that the strong OV copy z-scores are an artefact of picking S from the difference structure. For the key heads I compute the same 1-d copy score sᵀ·ΔOV·s for each of the first 6 principal read/write pairs. Only pairs 1–2 have read/write overlap above chance. If pairs 3–6, which are selected the same way but are not shared token directions, give similar |z|, the effect is a selection artefact. If only d1/d2 give strong negative values, the B-copies-more result is specific to the answer-token subspace.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
def dW(L,t): return W(A,L,t)-W(B,L,t)
R = np.vstack([dW(0,"self_attn.q_proj"), dW(0,"self_attn.k_proj"), dW(0,"self_attn.v_proj"), dW(0,"mlp.gate_proj"), dW(0,"mlp.up_proj")])
Vr=np.linalg.svd(R,full_matrices=False)[2][:20].T
Wm = np.hstack([dW(L,"mlp.down_proj") for L in [15,16,17]]+[dW(L,"self_attn.o_proj") for L in [12,13,14,15]])
Uw=np.linalg.svd(Wm,full_matrices=False)[0][:,:20]
P,s,Qt=np.linalg.svd(Vr.T@Uw); pin=Vr@P[:,:8]; pout=Uw@Qt.T[:,:8]
print("principal cos", s[:8].round(3))
D=[(pin[:,i]+pout[:,i])/np.linalg.norm(pin[:,i]+pout[:,i]) for i in range(8)]
rng=np.random.default_rng(0); rv=rng.standard_normal((640,400)); rv/=np.linalg.norm(rv,axis=0)
heads=[(13,1),(13,2),(15,0),(15,1),(16,1),(16,2),(17,0),(17,3),(14,0),(17,1)]
for (L,h) in heads:
    sl=slice(h*hd,(h+1)*hd)
    Ma=W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"); Mb=W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj")
    dM=Ma-Mb; null=np.einsum('ij,ik,kj->j',rv,dM,rv); mu,sd=null.mean(),null.std()
    # also in-only and out-only vectors
    zs=[(d@dM@d-mu)/sd for d in D]
    zin=[(pin[:,i]@dM@pin[:,i]-mu)/sd for i in range(4)]
    zout=[(pout[:,i]@dM@pout[:,i]-mu)/sd for i in range(4)]
    print(L,h,"pairs:", " ".join("%+.1f"%x for x in zs), "| in-only", " ".join("%+.1f"%x for x in zin), "| out-only", " ".join("%+.1f"%x for x in zout))
```